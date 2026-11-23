"""Train Exp034E from the Exp034A one-step latent bridge checkpoint."""
from __future__ import annotations

import argparse
import csv
import gc
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
from models.exp034e_mamba_unet3d import MambaLowResUNet3D
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp033i_bsfe_ada_gl_table_full import decode_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--init-ckpt', type=Path, required=True)
    p.add_argument('--smoke-test', action='store_true')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_csv(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(header)


def append_csv(path, row):
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


def mri_endpoint(mri, latent_channels):
    return F.avg_pool3d(mri, kernel_size=4, stride=4).repeat(1, latent_channels, 1, 1, 1)

def gradient_l1(pred, target):
    return sum(F.l1_loss(pred.diff(dim=a), target.diff(dim=a)) for a in (2, 3, 4)) / 3.0

def global_l1(pred, target):
    return F.l1_loss(F.avg_pool3d(pred, 5, stride=2, padding=2), F.avg_pool3d(target, 5, stride=2, padding=2))


def decode(ae, z, image_shape, latent_coords, latent_patch_size, latent_full_size):
    return decode_pred(ae, z, image_shape, latent_coords, latent_patch_size, latent_full_size)


def load_exp034a_weights(model, path):
    state = torch.load(path, map_location='cpu')
    source = state.get('model_state_dict', state)
    target = model.state_dict()
    matched = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    missing = [k for k in target if k not in matched]
    model.load_state_dict(matched, strict=False)
    del state, source, matched
    gc.collect()
    return len(target) - len(missing), len(missing)


def step(model, ae, mri_ae, batch, cfg, device, latent_coords, latent_patch_size, latent_full_size):
    pet = batch['pet'].to(device, non_blocking=True).float()
    mri = batch['mri'].to(device, non_blocking=True).float()
    with torch.no_grad(): x1 = mri_ae.encode(mri)
    t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
    z_pred = model(x1, t)
    recon = decode(ae, z_pred, pet.shape, latent_coords, latent_patch_size, latent_full_size)
    image = F.l1_loss(recon, pet); glob = global_l1(recon, pet); local = gradient_l1(recon, pet); lc = cfg['loss']
    return lc['image_l1_weight']*image + lc['global_l1_weight']*glob + lc['local_gradient_l1_weight']*local, {'image_l1': image, 'global_l1': glob, 'local_gradient_l1': local}, recon, pet


@torch.no_grad()
def evaluate(model, ae, mri_ae, loader, cfg, device, latent_coords, latent_patch_size, latent_full_size, limit=None):
    model.eval()
    rows = []
    for i, batch in enumerate(loader):
        _, terms, recon, pet = step(model, ae, mri_ae, batch, cfg, device, latent_coords, latent_patch_size, latent_full_size)
        mask = batch['brain_mask'].to(device, non_blocking=True).float()
        metrics = compute_image_metrics_detailed(recon, pet, mask=mask, data_range=2.0)
        rows.append({'image_l1': float(terms['image_l1']), 'global_l1': float(terms['global_l1']), 'local_gradient_l1': float(terms['local_gradient_l1']), **{k: float(metrics[k]) for k in ('ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')}})
        if limit is not None and i + 1 >= limit:
            break
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


def main():
    args = parse_args()
    with args.config.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda'))
    out = REPO_ROOT / cfg['experiment']['output_dir']
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    image_size = tuple(cfg['data']['image_size'])
    patch_size = tuple(cfg['data']['patch_size'])
    latent_full = tuple(v // 4 for v in image_size)
    latent_patch = tuple(v // 4 for v in patch_size)
    latent_coords = build_patch_grid(latent_full, latent_patch)

    train_ds = build_dataset(cfg, 'train')
    nw = int(cfg['training'].get('num_workers', 0))
    common = dict(num_workers=nw, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=nw > 0)
    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **common)
    val_sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, **common)
    small_loader = DataLoader(Subset(val_ds, range(min(len(val_ds), int(cfg['evaluation'].get('small_val_subjects', 4))))), batch_size=int(cfg['training']['batch_size']), shuffle=False)

    ae = load_autoencoder(cfg['ae']['kind'], REPO_ROOT / cfg['ae']['checkpoint_path'], device)
    mri_ae = load_autoencoder(cfg['mri_ae']['kind'], REPO_ROOT / cfg['mri_ae']['checkpoint_path'], device)
    for module in (ae, mri_ae):
        for p in module.parameters(): p.requires_grad_(False)
    m = cfg['model']
    model = MambaLowResUNet3D(
        in_channels=int(m['latent_channels']),
        out_channels=int(m['latent_channels']),
        channels=tuple(m['channels']),
        time_dim=int(m['time_dim']),
        norm_groups=int(m['norm_groups']),
        dropout=float(m.get('dropout', 0.0)),
        latent_shape=tuple(m['latent_shape']),
        mamba_token_dim=int(m.get('mamba_token_dim', 128)),
        mamba_d_state=int(m.get('mamba_d_state', 8)),
        mamba_d_conv=int(m.get('mamba_d_conv', 3)),
        mamba_expand=int(m.get('mamba_expand', 1)),
        mamba_line_chunk_size=int(m.get('mamba_line_chunk_size', 512)),
        mamba_depth=int(m.get('mamba_depth', 1)),
    )
    loaded, missing = load_exp034a_weights(model, REPO_ROOT / args.init_ckpt)
    model = model.to(device)
    base_params, mamba_params = [], []
    for name, parameter in model.named_parameters():
        (mamba_params if name.startswith('mamba_') else base_params).append(parameter)
    opt = torch.optim.AdamW(
        [
            {'params': base_params, 'lr': float(cfg['training'].get('backbone_learning_rate', 1e-5))},
            {'params': mamba_params, 'lr': float(cfg['training'].get('mamba_learning_rate', cfg['training']['learning_rate']))},
        ],
        weight_decay=float(cfg['training'].get('weight_decay', 1e-4)),
        betas=tuple(cfg['training'].get('betas', [0.9, 0.999])),
    )
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    amp = cfg['training'].get('amp', 'bf16')
    amp_dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    train_csv = out / 'logs/train_log.csv'
    val_csv = out / 'logs/val_log.csv'
    ensure_csv(train_csv, ['epoch', 'loss_total', 'image_l1', 'global_l1', 'local_gradient_l1', 'lr_backbone', 'lr_mamba', 'peak_gpu_mem_mb', 'epoch_time_sec'])
    ensure_csv(val_csv, ['epoch', 'image_l1', 'global_l1', 'local_gradient_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae'])
    warmup_epochs = int(cfg['training'].get('mamba_warmup_epochs', 0))
    print(json.dumps({
        'experiment': cfg['experiment']['name'],
        'loaded_exp034a_tensors': loaded,
        'new_tensors': missing,
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'mamba_warmup_epochs': warmup_epochs,
        'mri_ae_checkpoint': cfg['mri_ae']['checkpoint_path'], 'guide_loss_weights': cfg['loss'],
    }), flush=True)

    accum = max(int(cfg['training'].get('grad_accum', 1)), 1)
    best = -float('inf')
    for epoch in range(1, epochs + 1):
        train_backbone = epoch > warmup_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(train_backbone or name.startswith('mamba_'))
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []; terms_list = []
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        for i, batch in enumerate(train_loader):
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp != 'off'):
                loss, terms, _, _ = step(model, ae, mri_ae, batch, cfg, device, latent_coords, latent_patch, latent_full)
                scaled = loss / accum
            scaler.scale(scaled).backward()
            if (i + 1) % accum == 0 or i + 1 == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            terms_list.append({k: float(v.detach()) for k, v in terms.items()})
            if args.smoke_test:
                break
        scheduler.step()
        small = evaluate(model, ae, mri_ae, small_loader, cfg, device, latent_coords, latent_patch, latent_full, limit=1 if args.smoke_test else None)
        full = evaluate(model, ae, mri_ae, val_loader, cfg, device, latent_coords, latent_patch, latent_full, limit=1 if args.smoke_test else None)
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        avg = float(np.mean(losses))
        mt = {k: float(np.mean([x[k] for x in terms_list])) for k in terms_list[0]}
        append_csv(train_csv, [epoch, avg, mt['image_l1'], mt['global_l1'], mt['local_gradient_l1'], opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], peak, time.time() - started])
        append_csv(val_csv, [epoch, *[full[k] for k in ('image_l1', 'global_l1', 'local_gradient_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')]])
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_ssim': best,
            'metrics': {'small': small, 'full': full},
            'config': cfg,
        }
        torch.save(state, out / 'checkpoints/last.pt')
        if full['ssim'] > best:
            best = full['ssim']
            state['best_ssim'] = best
            torch.save(state, out / 'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch': epoch, 'backbone_trainable': train_backbone, 'train_image_l1': avg, 'small': small, 'full': full, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__':
    main()
