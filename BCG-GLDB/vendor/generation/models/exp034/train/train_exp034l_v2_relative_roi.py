"""Train Exp034L_v2: scale-stable relative Parc98 uptake supervision."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp026.utils.patch_codec import build_patch_grid
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034g_bsfe_table_mamba_unet3d import BSFETableMambaUNet3D
from train.train_bsfe_table_jointprior_latent_bridge_t1_to_fdg import infer_feature_type_ids
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv, mri_endpoint
from utils.ssim import SSIM3D


def regional_profile_losses(prediction, target, labels, brain_mask, num_regions, min_voxels):
    """Compare per-subject regional uptake profiles after robust global normalization."""
    profile_losses, contrast_losses = [], []
    for batch_index in range(prediction.shape[0]):
        region = labels[batch_index, 0].long().reshape(-1)
        valid = (region > 0) & (region <= int(num_regions)) & (brain_mask[batch_index, 0].reshape(-1) > 0.5)
        if not valid.any():
            continue
        region = region[valid]
        pred = prediction[batch_index, 0].reshape(-1)[valid]
        truth = target[batch_index, 0].reshape(-1)[valid]
        counts = pred.new_zeros(int(num_regions) + 1)
        pred_sums = pred.new_zeros(int(num_regions) + 1)
        truth_sums = truth.new_zeros(int(num_regions) + 1)
        counts.scatter_add_(0, region, torch.ones_like(pred))
        pred_sums.scatter_add_(0, region, pred)
        truth_sums.scatter_add_(0, region, truth)
        present = counts[1:] >= int(min_voxels)
        if present.any():
            pred_means = pred_sums[1:][present] / counts[1:][present]
            truth_means = truth_sums[1:][present] / counts[1:][present]
            # The cached targets are in [-1, 1]. Clipping regional means prevents
            # a few early-training outliers from dominating the regional signal.
            pred_means = pred_means.clamp(-1.0, 1.0)
            truth_means = truth_means.clamp(-1.0, 1.0)
            # Keep the regional terms on an image-loss-like scale for low-uptake ROIs.
            global_mean = truth.abs().mean().detach().clamp_min(0.25)
            weights = counts[1:][present].sqrt()
            weights = weights / weights.sum().clamp_min(1.0)
            profile_losses.append((F.smooth_l1_loss(pred_means / global_mean, truth_means / global_mean, reduction='none') * weights).sum())
            pred_contrast = (pred_means - pred_means.mean()) / global_mean
            truth_contrast = (truth_means - truth_means.mean()) / global_mean
            contrast_losses.append((F.smooth_l1_loss(pred_contrast, truth_contrast, reduction='none') * weights).sum())
    zero = prediction.sum() * 0.0
    return (torch.stack(profile_losses).mean() if profile_losses else zero,
            torch.stack(contrast_losses).mean() if contrast_losses else zero)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--init-ckpt', required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out = REPO_ROOT / cfg['experiment']['output_dir']
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    image_size, patch_size = tuple(cfg['data']['image_size']), tuple(cfg['data']['patch_size'])
    latent_full, latent_patch = tuple(v // 4 for v in image_size), tuple(v // 4 for v in patch_size)
    latent_coords = build_patch_grid(latent_full, latent_patch)
    train_ds = build_dataset(cfg, 'train')
    workers = int(cfg['training'].get('num_workers', 0))
    common = dict(num_workers=workers, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **common)
    val_sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, **common)
    small_loader = DataLoader(Subset(val_ds, range(min(len(val_ds), int(cfg['evaluation']['small_val_subjects'])))), batch_size=int(cfg['training']['batch_size']))

    ae = load_autoencoder(cfg['ae']['kind'], REPO_ROOT / cfg['ae']['checkpoint_path'], device)
    for parameter in ae.parameters(): parameter.requires_grad_(False)
    bsfe, head = build_bsfe_models(cfg, latent_full, device)
    bsfe_state = torch.load(REPO_ROOT / cfg['bsfe']['checkpoint_path'], map_location=device, weights_only=False)
    bsfe.load_state_dict(bsfe_state['bsfe']); head.load_state_dict(bsfe_state['head'])
    del bsfe_state, head
    for parameter in bsfe.parameters(): parameter.requires_grad_(False)

    feature_ids, feature_map = infer_feature_type_ids(train_ds.table_cols)
    m, tb = cfg['model'], cfg['table']
    model = BSFETableMambaUNet3D(
        table_num_features=len(train_ds.table_cols), feature_type_ids=feature_ids,
        in_channels=int(m['latent_channels']), out_channels=int(m['latent_channels']), channels=tuple(m['channels']),
        bsfe_channels=tuple(m['bsfe_channels']), time_dim=int(m['time_dim']), table_token_dim=int(tb['token_dim']),
        table_heads=int(tb['heads']), table_token_grids=tb['token_grids'], norm_groups=int(m['norm_groups']),
        dropout=float(m.get('dropout', 0.0)), table_lambda_init=float(tb['lambda_init']),
        latent_shape=tuple(m['latent_shape']), mamba_token_dim=int(m['mamba_token_dim']),
        mamba_depth=int(m['mamba_depth']), mamba_d_state=int(m['mamba_d_state']),
        mamba_d_conv=int(m['mamba_d_conv']), mamba_expand=int(m['mamba_expand']),
        mamba_line_chunk_size=int(m['mamba_line_chunk_size']),
    )
    initial = torch.load(REPO_ROOT / args.init_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    target = model.state_dict()
    compatible = {key: value for key, value in initial.items() if key in target and target[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    loaded, new = len(compatible), len(target) - len(compatible)
    del initial, compatible
    model = model.to(device)
    backbone, mamba = [], []
    for name, parameter in model.named_parameters():
        (mamba if name.startswith('mamba_') else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': float(cfg['training']['learning_rate'])},
        {'params': mamba, 'lr': float(cfg['training'].get('mamba_learning_rate', cfg['training']['learning_rate']))},
    ], weight_decay=float(cfg['training']['weight_decay']))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = cfg['training'].get('amp', 'bf16'); dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    loss_cfg = cfg['loss']
    ssim_loss_fn = SSIM3D(data_range=float(loss_cfg['data_range'])).to(device)
    ensure_csv(out / 'logs/train_log.csv', ['epoch', 'backbone_trainable', 'loss_total', 'loss_global_l1', 'loss_brain_l1', 'loss_ssim', 'loss_mse', 'loss_roi_profile', 'loss_roi_contrast', 'lr_backbone', 'lr_mamba', 'peak_gpu_mem_mb', 'epoch_time_sec'])
    ensure_csv(out / 'logs/val_log.csv', ['epoch', 'image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae'])
    warmup = int(cfg['training'].get('mamba_warmup_epochs', 0))
    print(json.dumps({'experiment': cfg['experiment']['name'], 'loaded_exp034g_tensors': loaded, 'new_tensors': new, 'parameters': sum(p.numel() for p in model.parameters()), 'table_features': len(train_ds.table_cols), 'feature_type_map': feature_map, 'mamba_warmup_epochs': warmup, 'loss_weights': loss_cfg}), flush=True)

    def run(batch):
        pet = batch['pet'].to(device, non_blocking=True).float()
        mri = batch['mri'].to(device, non_blocking=True).float()
        table = batch['table'].to(device, non_blocking=True).float()
        table_mask = batch['table_mask'].to(device, non_blocking=True).float()
        t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        with torch.no_grad(): pyramid = bsfe(mri)['pyramid']
        z = model(mri_endpoint(mri, int(m['latent_channels'])), t, pyramid, table, table_mask)
        recon = decode(ae, z, pet.shape, latent_coords, latent_patch, latent_full)
        brain_mask = batch['brain_mask'].to(device, non_blocking=True).float()
        parc98 = batch['parc98'].to(device, non_blocking=True)
        global_l1 = F.l1_loss(recon, pet)
        mask_sum = brain_mask.sum().clamp_min(1.0)
        brain_l1 = (torch.abs(recon - pet) * brain_mask).sum() / mask_sum
        ssim_loss = 1.0 - ssim_loss_fn(recon, pet)
        # MSE monotonically improves PSNR while avoiding the unstable log form of a direct PSNR loss.
        mse = F.mse_loss(recon, pet)
        roi_profile, roi_contrast = regional_profile_losses(recon, pet, parc98, brain_mask, int(loss_cfg['num_regions']), int(loss_cfg['min_region_voxels']))
        total = (float(loss_cfg['global_l1_weight']) * global_l1 + float(loss_cfg['brain_l1_weight']) * brain_l1
                 + float(loss_cfg['ssim_weight']) * ssim_loss + float(loss_cfg['mse_weight']) * mse
                 + float(loss_cfg['regional_profile_weight']) * roi_profile + float(loss_cfg['regional_contrast_weight']) * roi_contrast)
        return total, {'global_l1': global_l1, 'brain_l1': brain_l1, 'ssim': ssim_loss, 'mse': mse, 'roi_profile': roi_profile, 'roi_contrast': roi_contrast}, recon, pet

    @torch.no_grad()
    def evaluate(loader, limit=None):
        model.eval(); rows = []
        for index, batch in enumerate(loader):
            _, _, recon, pet = run(batch)
            metrics = compute_image_metrics_detailed(recon, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
            rows.append({'image_l1': float(F.l1_loss(recon, pet)), **{key: float(metrics[key]) for key in ('ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')}})
            if limit and index + 1 >= limit: break
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    accumulation = max(int(cfg['training'].get('grad_accum', 1)), 1); best = -float('inf')
    for epoch in range(1, epochs + 1):
        train_backbone = epoch > warmup
        for name, parameter in model.named_parameters(): parameter.requires_grad_(train_backbone or name.startswith('mamba_'))
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []; components = []; started = time.time(); torch.cuda.reset_peak_memory_stats(device)
        for index, batch in enumerate(train_loader):
            with torch.autocast('cuda', dtype=dtype, enabled=amp != 'off'):
                loss, terms, _, _ = run(batch)
            scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            components.append({key: float(value.detach()) for key, value in terms.items()})
            if args.smoke_test: break
        scheduler.step(); avg = float(np.mean(losses)); peak = torch.cuda.max_memory_allocated(device) / 1024**2
        small = evaluate(small_loader, 1 if args.smoke_test else None); full = evaluate(val_loader, 1 if args.smoke_test else None)
        mean_terms = {key: float(np.mean([row[key] for row in components])) for key in components[0]}
        append_csv(out / 'logs/train_log.csv', [epoch, train_backbone, avg, mean_terms['global_l1'], mean_terms['brain_l1'], mean_terms['ssim'], mean_terms['mse'], mean_terms['roi_profile'], mean_terms['roi_contrast'], optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], peak, time.time() - started])
        append_csv(out / 'logs/val_log.csv', [epoch, *[full[key] for key in ('image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')]])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_ssim': best, 'metrics': {'small': small, 'full': full}, 'config': cfg}
        torch.save(state, out / 'checkpoints/last.pt')
        if full['ssim'] > best:
            best = full['ssim']; state['best_ssim'] = best; torch.save(state, out / 'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch': epoch, 'backbone_trainable': train_backbone, 'train_loss': avg, 'loss_terms': mean_terms, 'full': full, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__': main()
