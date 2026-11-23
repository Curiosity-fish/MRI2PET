"""Train Exp034F: frozen BSFE controls plus high-resolution table BiCA."""
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
from models.exp034f_bsfe_table_highres_unet3d import BSFETableHighResUNet3D
from train.train_bsfe_table_jointprior_latent_bridge_t1_to_fdg import infer_feature_type_ids
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv, mri_endpoint


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
    model = BSFETableHighResUNet3D(
        table_num_features=len(train_ds.table_cols), feature_type_ids=feature_ids,
        in_channels=int(m['latent_channels']), out_channels=int(m['latent_channels']), channels=tuple(m['channels']),
        bsfe_channels=tuple(m['bsfe_channels']), time_dim=int(m['time_dim']), table_token_dim=int(tb['token_dim']),
        table_heads=int(tb['heads']), table_token_grids=tb['token_grids'], norm_groups=int(m['norm_groups']),
        dropout=float(m.get('dropout', 0.0)), table_lambda_init=float(tb['lambda_init']),
    )
    initial = torch.load(REPO_ROOT / args.init_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    target = model.state_dict()
    compatible = {key: value for key, value in initial.items() if key in target and target[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    loaded, new = len(compatible), len(target) - len(compatible)
    del initial, compatible
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = cfg['training'].get('amp', 'bf16'); dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    ensure_csv(out / 'logs/train_log.csv', ['epoch', 'image_l1', 'lr', 'peak_gpu_mem_mb', 'epoch_time_sec'])
    ensure_csv(out / 'logs/val_log.csv', ['epoch', 'image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae'])
    print(json.dumps({'experiment': cfg['experiment']['name'], 'loaded_exp034a_tensors': loaded, 'new_tensors': new, 'parameters': sum(p.numel() for p in model.parameters()), 'table_features': len(train_ds.table_cols), 'feature_type_map': feature_map}), flush=True)

    def run(batch):
        pet = batch['pet'].to(device, non_blocking=True).float()
        mri = batch['mri'].to(device, non_blocking=True).float()
        table = batch['table'].to(device, non_blocking=True).float()
        table_mask = batch['table_mask'].to(device, non_blocking=True).float()
        t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        with torch.no_grad(): pyramid = bsfe(mri)['pyramid']
        z = model(mri_endpoint(mri, int(m['latent_channels'])), t, pyramid, table, table_mask)
        recon = decode(ae, z, pet.shape, latent_coords, latent_patch, latent_full)
        return F.l1_loss(recon, pet), recon, pet

    @torch.no_grad()
    def evaluate(loader, limit=None):
        model.eval(); rows = []
        for index, batch in enumerate(loader):
            loss, recon, pet = run(batch)
            metrics = compute_image_metrics_detailed(recon, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
            rows.append({'image_l1': float(loss), **{key: float(metrics[key]) for key in ('ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')}})
            if limit and index + 1 >= limit: break
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    accumulation = max(int(cfg['training'].get('grad_accum', 1)), 1); best = -float('inf')
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []; started = time.time(); torch.cuda.reset_peak_memory_stats(device)
        for index, batch in enumerate(train_loader):
            with torch.autocast('cuda', dtype=dtype, enabled=amp != 'off'):
                loss, _, _ = run(batch)
            scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if args.smoke_test: break
        scheduler.step(); avg = float(np.mean(losses)); peak = torch.cuda.max_memory_allocated(device) / 1024**2
        small = evaluate(small_loader, 1 if args.smoke_test else None); full = evaluate(val_loader, 1 if args.smoke_test else None)
        append_csv(out / 'logs/train_log.csv', [epoch, avg, optimizer.param_groups[0]['lr'], peak, time.time() - started])
        append_csv(out / 'logs/val_log.csv', [epoch, *[full[key] for key in ('image_l1', 'ssim', 'psnr', 'mae', 'rmse', 'brain_ssim', 'brain_psnr', 'brain_mae')]])
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_ssim': best, 'metrics': {'small': small, 'full': full}, 'config': cfg}
        torch.save(state, out / 'checkpoints/last.pt')
        if full['ssim'] > best:
            best = full['ssim']; state['best_ssim'] = best; torch.save(state, out / 'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch': epoch, 'train_image_l1': avg, 'full': full, 'peak_gpu_mem_mb': peak, 'best_ssim': best}), flush=True)


if __name__ == '__main__': main()
