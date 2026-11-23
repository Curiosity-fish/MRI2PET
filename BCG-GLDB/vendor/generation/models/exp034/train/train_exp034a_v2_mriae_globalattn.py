"""Train Exp034A_v2: frozen MRI-AE endpoint and global-attention residual U-Net."""
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
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034a_v2_global_attention_unet3d import Exp034AV2GlobalAttentionUNet3D
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(int(cfg['experiment']['seed']))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out = ROOT / cfg['experiment']['output_dir']
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    image_size, patch_size = tuple(cfg['data']['image_size']), tuple(cfg['data']['patch_size'])
    latent_full, latent_patch = tuple(v // 4 for v in image_size), tuple(v // 4 for v in patch_size)
    coords = build_patch_grid(latent_full, latent_patch)
    train_ds = build_dataset(cfg, 'train')
    workers = int(cfg['training'].get('num_workers', 0))
    loader_kwargs = dict(num_workers=workers, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **loader_kwargs)
    val_sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)
    small_loader = DataLoader(Subset(val_ds, range(min(len(val_ds), int(cfg['evaluation']['small_val_subjects'])))), batch_size=1)
    pet_ae = load_autoencoder(cfg['ae']['kind'], ROOT / cfg['ae']['checkpoint_path'], device)
    mri_ae = load_autoencoder(cfg['mri_ae']['kind'], ROOT / cfg['mri_ae']['checkpoint_path'], device)
    for module in (pet_ae, mri_ae):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model_cfg = cfg['model']
    model = Exp034AV2GlobalAttentionUNet3D(
        in_channels=int(model_cfg['latent_channels']), out_channels=int(model_cfg['latent_channels']),
        channels=tuple(model_cfg['channels']), time_dim=int(model_cfg['time_dim']),
        norm_groups=int(model_cfg['norm_groups']), attention_heads=int(model_cfg['attention_heads']),
        dropout=float(model_cfg.get('dropout', 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = cfg['training'].get('amp', 'bf16')
    dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    ensure_csv(out / 'logs/train_log.csv', ['epoch', 'image_l1', 'lr', 'peak_gpu_mem_mb', 'epoch_time_sec'])
    ensure_csv(out / 'logs/val_log.csv', ['epoch', 'image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae'])
    print(json.dumps({'experiment': cfg['experiment']['name'], 'parameters': sum(p.numel() for p in model.parameters()), 'mri_ae_frozen': True, 'pet_ae_frozen': True, 'attention': 'exact_global_3d_at_all_encoder_decoder_stages', 'latent_shape': [4, *latent_full]}), flush=True)

    def run(batch):
        pet, mri = batch['pet'].to(device, non_blocking=True).float(), batch['mri'].to(device, non_blocking=True).float()
        with torch.no_grad():
            x1 = mri_ae.encode(mri)
        if tuple(x1.shape[1:]) != (4, *latent_full):
            raise RuntimeError(f'Unexpected MRI-AE latent shape: {tuple(x1.shape)}')
        t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        z = model(x1, t)
        recon = decode(pet_ae, z, pet.shape, coords, latent_patch, latent_full)
        return F.l1_loss(recon, pet), recon, pet

    @torch.no_grad()
    def evaluate(loader, limit=None):
        model.eval(); rows = []
        for index, batch in enumerate(loader):
            with torch.autocast('cuda', dtype=dtype, enabled=amp != 'off'):
                loss, recon, pet = run(batch)
            metrics = compute_image_metrics_detailed(recon, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
            rows.append({'image_l1': float(loss), **{key: float(metrics[key]) for key in ('ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')}})
            if limit and index + 1 >= limit:
                break
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    best = -float('inf'); accumulation = max(int(cfg['training']['grad_accum']), 1)
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []; started = time.time(); torch.cuda.reset_peak_memory_stats(device)
        for index, batch in enumerate(train_loader):
            with torch.autocast('cuda', dtype=dtype, enabled=amp != 'off'):
                loss, _, _ = run(batch)
            scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if args.smoke_test:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                break
        scheduler.step(); peak = torch.cuda.max_memory_allocated(device) / 1024**2
        full = evaluate(val_loader, 1 if args.smoke_test else None)
        append_csv(out / 'logs/train_log.csv', [epoch, float(np.mean(losses)), optimizer.param_groups[0]['lr'], peak, time.time() - started])
        append_csv(out / 'logs/val_log.csv', [epoch, *[full[key] for key in ('image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')]])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_ssim': best, 'metrics': {'full': full}, 'config': cfg}
        torch.save(state, out / 'checkpoints/last.pt')
        if full['ssim'] > best:
            best = full['ssim']; state['best_ssim'] = best; torch.save(state, out / 'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch': epoch, 'train_image_l1': float(np.mean(losses)), 'full': full, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__':
    main()
