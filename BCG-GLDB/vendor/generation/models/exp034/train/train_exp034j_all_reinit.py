"""Exp034J: randomize every trainable module while retaining a frozen PET AE encoder."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp026.utils.patch_codec import build_patch_grid
from exp032.core import PatchBackboneDecoder
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034g_bsfe_table_mamba_unet3d import BSFETableMambaUNet3D
from train.train_bsfe_table_jointprior_latent_bridge_t1_to_fdg import infer_feature_type_ids
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv, mri_endpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out = ROOT / cfg['experiment']['output_dir']
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)

    image_size, patch_size = tuple(cfg['data']['image_size']), tuple(cfg['data']['patch_size'])
    latent_full = tuple(value // 4 for value in image_size)
    latent_patch = tuple(value // 4 for value in patch_size)
    coords = build_patch_grid(latent_full, latent_patch)
    train_ds = build_dataset(cfg, 'train')
    workers = int(cfg['training'].get('num_workers', 0))
    loader_args = dict(num_workers=workers, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **loader_args)
    val_sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_args)

    # Encoder weights are the sole retained initialization; all other modules start randomly.
    ae = load_autoencoder(cfg['ae']['kind'], ROOT / cfg['ae']['checkpoint_path'], device)
    ae.decoder = PatchBackboneDecoder(ae.image_channels, ae.latent_channels, ae.base_channels, ae.norm_groups, ae.dropout).to(device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    for parameter in ae.decoder.parameters():
        parameter.requires_grad_(True)

    bsfe, head = build_bsfe_models(cfg, latent_full, device)
    for module in (bsfe, head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    feature_ids, _ = infer_feature_type_ids(train_ds.table_cols)
    model_cfg, table_cfg = cfg['model'], cfg['table']
    model = BSFETableMambaUNet3D(
        table_num_features=len(train_ds.table_cols), feature_type_ids=feature_ids, in_channels=4, out_channels=4,
        channels=tuple(model_cfg['channels']), bsfe_channels=tuple(model_cfg['bsfe_channels']), time_dim=int(model_cfg['time_dim']),
        table_token_dim=int(table_cfg['token_dim']), table_heads=int(table_cfg['heads']), table_token_grids=table_cfg['token_grids'],
        norm_groups=int(model_cfg['norm_groups']), dropout=float(model_cfg['dropout']), table_lambda_init=float(table_cfg['lambda_init']),
        latent_shape=tuple(model_cfg['latent_shape']), mamba_token_dim=int(model_cfg['mamba_token_dim']),
        mamba_depth=int(model_cfg['mamba_depth']), mamba_d_state=int(model_cfg['mamba_d_state']), mamba_d_conv=int(model_cfg['mamba_d_conv']),
        mamba_expand=int(model_cfg['mamba_expand']), mamba_line_chunk_size=int(model_cfg['mamba_line_chunk_size'])).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': float(cfg['training']['learning_rate'])},
        {'params': bsfe.parameters(), 'lr': float(cfg['training']['bsfe_learning_rate'])},
        {'params': head.parameters(), 'lr': float(cfg['training']['bsfe_learning_rate'])},
        {'params': ae.decoder.parameters(), 'lr': float(cfg['training']['decoder_learning_rate'])},
    ], weight_decay=float(cfg['training']['weight_decay']))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = cfg['training'].get('amp', 'bf16')
    amp_dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    ensure_csv(out / 'logs/train_log.csv', ['epoch', 'image_l1', 'lr_model', 'lr_bsfe', 'lr_decoder', 'peak_gpu_mem_mb', 'epoch_time_sec'])
    ensure_csv(out / 'logs/val_log.csv', ['epoch', 'image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae'])
    print(json.dumps({'experiment': cfg['experiment']['name'], 'initialization': 'random_except_ae_encoder', 'ae_encoder_trainable': False, 'trainable_parameters': sum(p.numel() for g in optimizer.param_groups for p in g['params'])}), flush=True)

    def run(batch):
        pet, mri = batch['pet'].to(device).float(), batch['mri'].to(device).float()
        table, mask = batch['table'].to(device).float(), batch['table_mask'].to(device).float()
        timestep = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        pyramid = bsfe(mri)['pyramid']
        latent = model(mri_endpoint(mri, 4), timestep, pyramid, table, mask)
        reconstruction = decode(ae, latent, pet.shape, coords, latent_patch, latent_full)
        return F.l1_loss(reconstruction, pet), reconstruction, pet

    @torch.no_grad()
    def evaluate(loader, limit=None):
        model.eval(); bsfe.eval(); head.eval(); ae.decoder.eval()
        rows = []
        for index, batch in enumerate(loader):
            loss, reconstruction, pet = run(batch)
            metrics = compute_image_metrics_detailed(reconstruction, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
            rows.append({'image_l1': float(loss), **{key: float(metrics[key]) for key in ('ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')}})
            if limit and index + 1 >= limit:
                break
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    accumulation = max(int(cfg['training'].get('grad_accum', 1)), 1)
    best = -float('inf')
    for epoch in range(1, epochs + 1):
        model.train(); bsfe.train(); head.train(); ae.decoder.train()
        optimizer.zero_grad(set_to_none=True)
        losses, start = [], time.time()
        torch.cuda.reset_peak_memory_stats(device)
        for index, batch in enumerate(train_loader):
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp != 'off'):
                loss, _, _ = run(batch)
            scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if args.smoke_test:
                break
        scheduler.step()
        metrics = evaluate(val_loader, 1 if args.smoke_test else None)
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        append_csv(out / 'logs/train_log.csv', [epoch, float(np.mean(losses)), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[3]['lr'], peak, time.time() - start])
        append_csv(out / 'logs/val_log.csv', [epoch, *[metrics[key] for key in ('image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')]])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'bsfe_state_dict': bsfe.state_dict(), 'head_state_dict': head.state_dict(), 'decoder_state_dict': ae.decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_ssim': best, 'metrics': {'full': metrics}, 'config': cfg}
        torch.save(state, out / 'checkpoints/last.pt')
        if metrics['ssim'] > best:
            best = metrics['ssim']; state['best_ssim'] = best; torch.save(state, out / 'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch': epoch, 'train_image_l1': float(np.mean(losses)), 'full': metrics, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__':
    main()
