"""Exp034A: MRI reshape/pool endpoint -> PET latent with a basic 3D U-Net."""
from __future__ import annotations

import argparse
import csv
import json
import random
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
from models.exp034a_basic_unet3d import BasicUNet3D
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp033i_bsfe_ada_gl_table_full import decode_pred


def parse_args():
    p = argparse.ArgumentParser(description='Exp034A basic MRI-pool to PET-latent one-step training')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke-test', action='store_true')
    return p.parse_args()


def load_config(path):
    with Path(path).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['experiment']['config_path'] = str(path)
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_csv(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='', encoding='utf-8') as f: csv.writer(f).writerow(header)


def append_csv(path, row):
    with path.open('a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)


def build_loaders(cfg):
    train_ds = build_dataset(cfg, 'train')
    nw = int(cfg['training'].get('num_workers', 0))
    common = dict(num_workers=nw, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=nw > 0)
    train = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **common)
    sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = sets[0] if len(sets) == 1 else ConcatDataset(sets)
    val = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, **common)
    small = DataLoader(Subset(val_ds, range(min(len(val_ds), int(cfg['evaluation'].get('small_val_subjects', 4))))), batch_size=int(cfg['training']['batch_size']), shuffle=False)
    return train, val, small


def mri_endpoint(mri, latent_channels):
    x = F.avg_pool3d(mri, kernel_size=4, stride=4)
    return x.repeat(1, latent_channels, 1, 1, 1)


def decode(ae, z, image_shape, latent_coords, latent_patch_size, latent_full_size):
    return decode_pred(ae, z, image_shape, latent_coords, latent_patch_size, latent_full_size)


def step(model, ae, batch, cfg, device, latent_coords, latent_patch_size, latent_full_size):
    pet = batch['pet'].to(device, non_blocking=True).float()
    mri = batch['mri'].to(device, non_blocking=True).float()
    x1 = mri_endpoint(mri, int(cfg['model']['latent_channels']))
    t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
    z_pred = model(x1, t)
    recon = decode(ae, z_pred, pet.shape, latent_coords, latent_patch_size, latent_full_size)
    return F.l1_loss(recon, pet), recon, pet


@torch.no_grad()
def evaluate(model, ae, loader, cfg, device, latent_coords, latent_patch_size, latent_full_size):
    model.eval(); rows = []
    for batch in loader:
        loss, recon, pet = step(model, ae, batch, cfg, device, latent_coords, latent_patch_size, latent_full_size)
        mask = batch['brain_mask'].to(device, non_blocking=True).float()
        m = compute_image_metrics_detailed(recon, pet, mask=mask, data_range=2.0)
        rows.append({'image_l1': float(loss), **{k: float(m[k]) for k in ('ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')}})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def checkpoint(path, epoch, model, optimizer, scheduler, best_ssim, cfg, metrics):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'best_ssim': best_ssim, 'config': cfg, 'metrics': metrics}, path)


def main():
    args = parse_args(); cfg = load_config(args.config); set_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out = REPO_ROOT / cfg['experiment']['output_dir']; (out/'checkpoints').mkdir(parents=True, exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    image_size = tuple(cfg['data']['image_size']); patch_size = tuple(cfg['data']['patch_size'])
    latent_full = tuple(v // 4 for v in image_size); latent_patch = tuple(v // 4 for v in patch_size)
    latent_coords = build_patch_grid(latent_full, latent_patch)
    train_loader, val_loader, small_loader = build_loaders(cfg)
    ae = load_autoencoder(cfg['ae']['kind'], REPO_ROOT/cfg['ae']['checkpoint_path'], device)
    for p in ae.parameters(): p.requires_grad_(False)
    m = cfg['model']; model = BasicUNet3D(in_channels=int(m['latent_channels']), out_channels=int(m['latent_channels']), channels=tuple(m['channels']), time_dim=int(m['time_dim']), norm_groups=int(m['norm_groups']), dropout=float(m.get('dropout', 0))).to(device)
    total = sum(p.numel() for p in model.parameters()); print(json.dumps({'experiment': cfg['experiment']['name'], 'parameters': total, 'train_samples': len(train_loader.dataset), 'val_samples': len(val_loader.dataset)}), flush=True)
    (out/'architecture.txt').write_text(json.dumps({'model': m, 'parameters': total, 'loss': 'image_l1_only'}, indent=2))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']), betas=tuple(cfg['training'].get('betas',[.9,.999])))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs']); warmup = int(cfg['training'].get('warmup_epochs', 0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1))
    scaler_enabled = cfg['training'].get('amp','bf16') == 'fp16'; scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    start, best = 1, -float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location=device); model.load_state_dict(state['model_state_dict']); opt.load_state_dict(state['optimizer_state_dict']); sched.load_state_dict(state['scheduler_state_dict']); start = int(state['epoch']) + 1; best = float(state.get('best_ssim', best))
    train_csv, val_csv = out/'logs/train_log.csv', out/'logs/val_log.csv'
    ensure_csv(train_csv, ['epoch','image_l1','lr','peak_gpu_mem_mb','epoch_time_sec'])
    ensure_csv(val_csv, ['epoch','image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae'])
    accum = max(int(cfg['training'].get('grad_accum', 1)), 1); amp_dtype = torch.bfloat16 if cfg['training'].get('amp','bf16') == 'bf16' else torch.float16
    for epoch in range(start, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True); losses = []; started = time.time(); torch.cuda.reset_peak_memory_stats(device)
        for i, batch in enumerate(train_loader):
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=cfg['training'].get('amp','bf16') != 'off'):
                loss, _, _ = step(model, ae, batch, cfg, device, latent_coords, latent_patch, latent_full)
                scaled = loss / accum
            scaler.scale(scaled).backward()
            if (i + 1) % accum == 0 or i + 1 == len(train_loader):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        sched.step()
        avg_loss = float(np.mean(losses)); peak = torch.cuda.max_memory_allocated(device) / 1024**2
        append_csv(train_csv, [epoch, avg_loss, opt.param_groups[0]['lr'], peak, time.time()-started])
        small_metrics = evaluate(model, ae, small_loader, cfg, device, latent_coords, latent_patch, latent_full)
        full_metrics = evaluate(model, ae, val_loader, cfg, device, latent_coords, latent_patch, latent_full)
        append_csv(val_csv, [epoch, *[full_metrics[k] for k in ['image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae']]])
        checkpoint(out/'checkpoints/last.pt', epoch, model, opt, sched, best, cfg, {'small': small_metrics, 'full': full_metrics})
        if full_metrics['ssim'] > best:
            best = full_metrics['ssim']; checkpoint(out/'checkpoints/best_ssim.pt', epoch, model, opt, sched, best, cfg, {'small': small_metrics, 'full': full_metrics})
        print(json.dumps({'epoch': epoch, 'train_image_l1': avg_loss, 'small': small_metrics, 'full': full_metrics, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__': main()
