"""Exp034C_v2: MRI-AE endpoint with high-resolution table interaction and table-alignment loss."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp026.utils.patch_codec import build_patch_grid
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034c_table_bica_unet3d import TableBiCAUNet3D
from train.train_bsfe_table_jointprior_latent_bridge_t1_to_fdg import infer_feature_type_ids
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv


class LatentTablePredictor(nn.Module):
    """Reads only the predicted PET latent, preventing table-input shortcut learning."""
    def __init__(self, latent_channels: int, table_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.LayerNorm(latent_channels), nn.Linear(latent_channels, latent_channels * 4), nn.SiLU(), nn.Linear(latent_channels * 4, table_features))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def masked_table_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    observed = (missing_mask < 0.5).float()
    return (F.smooth_l1_loss(prediction, target, reduction='none') * observed).sum() / observed.sum().clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out = ROOT / cfg['experiment']['output_dir']
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    image_size, patch_size = tuple(cfg['data']['image_size']), tuple(cfg['data']['patch_size'])
    latent_full, latent_patch = tuple(v // 4 for v in image_size), tuple(v // 4 for v in patch_size)
    coords = build_patch_grid(latent_full, latent_patch)
    train_ds = build_dataset(cfg, 'train')
    workers = int(cfg['training'].get('num_workers', 0))
    loader_args = dict(num_workers=workers, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **loader_args)
    val_sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_args)
    small_loader = DataLoader(Subset(val_ds, range(min(len(val_ds), int(cfg['evaluation']['small_val_subjects'])))), batch_size=1)
    pet_ae = load_autoencoder(cfg['ae']['kind'], ROOT / cfg['ae']['checkpoint_path'], device)
    mri_ae = load_autoencoder(cfg['mri_ae']['kind'], ROOT / cfg['mri_ae']['checkpoint_path'], device)
    for module in (pet_ae, mri_ae):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    feature_ids, feature_map = infer_feature_type_ids(train_ds.table_cols)
    model_cfg, table_cfg = cfg['model'], cfg['table']
    model = TableBiCAUNet3D(table_num_features=len(train_ds.table_cols), feature_type_ids=feature_ids, in_channels=int(model_cfg['latent_channels']), out_channels=int(model_cfg['latent_channels']), channels=tuple(model_cfg['channels']), time_dim=int(model_cfg['time_dim']), token_dim=int(table_cfg['token_dim']), token_grids=table_cfg['token_grids'], heads=int(table_cfg['heads']), norm_groups=int(model_cfg['norm_groups']), dropout=float(model_cfg['dropout']), lambda_init=float(table_cfg['lambda_init']), table_stage_indices=table_cfg['stage_indices']).to(device)
    table_head = LatentTablePredictor(int(model_cfg['latent_channels']), len(train_ds.table_cols)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(table_head.parameters()), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']), betas=tuple(cfg['training']['betas']))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = cfg['training'].get('amp', 'bf16'); amp_dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    ensure_csv(out / 'logs/train_log.csv', ['epoch', 'loss_total', 'loss_image_l1', 'loss_table_aux', 'lr', 'peak_gpu_mem_mb', 'epoch_time_sec'])
    ensure_csv(out / 'logs/val_log.csv', ['epoch', 'image_l1', 'table_aux', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae'])
    print(json.dumps({'experiment': cfg['experiment']['name'], 'mri_ae_checkpoint': cfg['mri_ae']['checkpoint_path'], 'mri_ae_frozen': True, 'table_stage_indices': table_cfg['stage_indices'], 'table_aux_weight': cfg['loss']['table_aux_weight'], 'parameters': sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in table_head.parameters()), 'feature_type_map': feature_map}), flush=True)

    def step(batch):
        pet, mri = batch['pet'].to(device).float(), batch['mri'].to(device).float()
        table, table_mask = batch['table'].to(device).float(), batch['table_mask'].to(device).float()
        with torch.no_grad():
            x1 = mri_ae.encode(mri)
        expected = (int(model_cfg['latent_channels']), *latent_full)
        if tuple(x1.shape[1:]) != expected:
            raise RuntimeError(f'Exp034B MRI-AE endpoint expected [B,{expected}], got {tuple(x1.shape)}')
        t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        z_pred = model(x1, t, table, table_mask)
        reconstruction = decode(pet_ae, z_pred, pet.shape, coords, latent_patch, latent_full)
        image_l1 = F.l1_loss(reconstruction, pet)
        table_aux = masked_table_smooth_l1(table_head(z_pred), table, table_mask)
        loss = float(cfg['loss']['image_l1_weight']) * image_l1 + float(cfg['loss']['table_aux_weight']) * table_aux
        return loss, {'image_l1': image_l1, 'table_aux': table_aux}, reconstruction, pet

    @torch.no_grad()
    def evaluate(loader, limit=None):
        model.eval(); table_head.eval(); rows = []
        for index, batch in enumerate(loader):
            _, terms, reconstruction, pet = step(batch)
            metrics = compute_image_metrics_detailed(reconstruction, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
            rows.append({'image_l1': float(terms['image_l1']), 'table_aux': float(terms['table_aux']), **{key: float(metrics[key]) for key in ('ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')}})
            if limit and index + 1 >= limit: break
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    accumulation = max(int(cfg['training']['grad_accum']), 1); best = -float('inf')
    for epoch in range(1, epochs + 1):
        model.train(); table_head.train(); optimizer.zero_grad(set_to_none=True); losses = []; components = []; started = time.time(); torch.cuda.reset_peak_memory_stats(device)
        for index, batch in enumerate(train_loader):
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp != 'off'):
                loss, terms, _, _ = step(batch)
            scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach())); components.append({key: float(value.detach()) for key, value in terms.items()})
            if args.smoke_test: break
        scheduler.step(); mean_terms = {key: float(np.mean([row[key] for row in components])) for key in components[0]}; peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        small = evaluate(small_loader, 1 if args.smoke_test else None); full = evaluate(val_loader, 1 if args.smoke_test else None)
        append_csv(out / 'logs/train_log.csv', [epoch, float(np.mean(losses)), mean_terms['image_l1'], mean_terms['table_aux'], optimizer.param_groups[0]['lr'], peak, time.time() - started])
        append_csv(out / 'logs/val_log.csv', [epoch, *[full[key] for key in ('image_l1', 'table_aux', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')]])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'table_head_state_dict': table_head.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_ssim': best, 'metrics': {'small': small, 'full': full}, 'config': cfg}
        torch.save(state, out / 'checkpoints/last.pt')
        if full['ssim'] > best:
            best = full['ssim']; state['best_ssim'] = best; torch.save(state, out / 'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'loss_terms': mean_terms, 'full': full, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__':
    main()
