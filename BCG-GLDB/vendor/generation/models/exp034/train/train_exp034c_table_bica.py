"""Exp034C: Exp034A with seven-stage image/table bidirectional attention."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.exp034c_table_bica_unet3d import TableBiCAUNet3D
from train.train_bsfe_table_jointprior_latent_bridge_t1_to_fdg import infer_feature_type_ids
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import (
    append_csv, build_patch_grid, checkpoint, compute_image_metrics_detailed,
    decode, ensure_csv, load_autoencoder, load_config, mri_endpoint, np,
    parse_args, random, set_seed, time, yaml,
)
from torch.utils.data import ConcatDataset, DataLoader, Subset


def build_loaders(cfg):
    train_ds = build_dataset(cfg, 'train')
    nw = int(cfg['training'].get('num_workers', 0))
    common = dict(num_workers=nw, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=nw > 0)
    train = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **common)
    sets = [build_dataset(cfg, split, train_dataset=train_ds) for split in cfg['evaluation']['val_splits']]
    val_ds = sets[0] if len(sets) == 1 else ConcatDataset(sets)
    val = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, **common)
    small = DataLoader(Subset(val_ds, range(min(len(val_ds), int(cfg['evaluation'].get('small_val_subjects', 4))))), batch_size=int(cfg['training']['batch_size']), shuffle=False)
    return train_ds, train, val, small


def step(model, ae, batch, cfg, device, latent_coords, latent_patch_size, latent_full_size):
    pet = batch['pet'].to(device, non_blocking=True).float()
    x1 = mri_endpoint(batch['mri'].to(device, non_blocking=True).float(), int(cfg['model']['latent_channels']))
    table = batch['table'].to(device, non_blocking=True).float()
    table_mask = batch['table_mask'].to(device, non_blocking=True).float()
    t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
    z_pred = model(x1, t, table, table_mask)
    recon = decode(ae, z_pred, pet.shape, latent_coords, latent_patch_size, latent_full_size)
    return F.l1_loss(recon, pet), recon, pet


@torch.no_grad()
def evaluate(model, ae, loader, cfg, device, latent_coords, latent_patch_size, latent_full_size, max_batches=None):
    model.eval(); rows = []
    for batch_idx, batch in enumerate(loader):
        loss, recon, pet = step(model, ae, batch, cfg, device, latent_coords, latent_patch_size, latent_full_size)
        m = compute_image_metrics_detailed(recon, pet, mask=batch['brain_mask'].to(device).float(), data_range=2.0)
        rows.append({'image_l1': float(loss), **{k: float(m[k]) for k in ('ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')}})
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    args = parse_args(); cfg = load_config(args.config); set_seed(int(cfg['experiment'].get('seed', 42)))
    device = torch.device(cfg['training'].get('device', 'cuda')); out = REPO_ROOT / cfg['experiment']['output_dir']
    for name in ('checkpoints', 'logs'): (out/name).mkdir(parents=True, exist_ok=True)
    image_size, patch_size = tuple(cfg['data']['image_size']), tuple(cfg['data']['patch_size'])
    latent_full, latent_patch = tuple(v//4 for v in image_size), tuple(v//4 for v in patch_size)
    latent_coords = build_patch_grid(latent_full, latent_patch)
    train_ds, train_loader, val_loader, small_loader = build_loaders(cfg)
    feature_ids, feature_map = infer_feature_type_ids(train_ds.table_cols)
    ae = load_autoencoder(cfg['ae']['kind'], REPO_ROOT/cfg['ae']['checkpoint_path'], device)
    for p in ae.parameters(): p.requires_grad_(False)
    m, tb = cfg['model'], cfg['table']
    model = TableBiCAUNet3D(table_num_features=len(train_ds.table_cols), feature_type_ids=feature_ids, in_channels=int(m['latent_channels']), out_channels=int(m['latent_channels']), channels=tuple(m['channels']), time_dim=int(m['time_dim']), token_dim=int(tb['token_dim']), token_grids=tb['token_grids'], heads=int(tb['heads']), norm_groups=int(m['norm_groups']), dropout=float(m.get('dropout', 0)), lambda_init=float(tb['lambda_init'])).to(device)
    total = sum(p.numel() for p in model.parameters())
    report = {'experiment':cfg['experiment']['name'], 'parameters':total, 'table_features':len(train_ds.table_cols), 'table_columns':train_ds.table_cols, 'feature_type_map':feature_map, 'train_samples':len(train_loader.dataset), 'val_samples':len(val_loader.dataset)}
    print(json.dumps(report), flush=True); (out/'architecture.txt').write_text(json.dumps({'model':m,'table':tb,'report':report,'loss':'image_l1_only'}, indent=2))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']), betas=tuple(cfg['training'].get('betas',[.9,.999])))
    epochs = 1 if args.smoke_test else int(cfg['training']['epochs']); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training'].get('amp','bf16') == 'fp16')
    train_csv, val_csv = out/'logs/train_log.csv', out/'logs/val_log.csv'; ensure_csv(train_csv,['epoch','image_l1','lr','peak_gpu_mem_mb','epoch_time_sec']); ensure_csv(val_csv,['epoch','image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae'])
    accum, best = max(int(cfg['training'].get('grad_accum',1)),1), -float('inf'); amp_dtype = torch.bfloat16 if cfg['training'].get('amp','bf16') == 'bf16' else torch.float16
    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); losses=[]; started=time.time(); torch.cuda.reset_peak_memory_stats(device)
        for i,batch in enumerate(train_loader):
            with torch.autocast('cuda', dtype=amp_dtype, enabled=cfg['training'].get('amp','bf16') != 'off'):
                loss,_,_ = step(model,ae,batch,cfg,device,latent_coords,latent_patch,latent_full); scaled=loss/accum
            scaler.scale(scaled).backward()
            if (i+1)%accum == 0 or i+1 == len(train_loader): scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if args.smoke_test:
                break
        sched.step(); avg=float(np.mean(losses)); peak=torch.cuda.max_memory_allocated(device)/1024**2
        append_csv(train_csv,[epoch,avg,opt.param_groups[0]['lr'],peak,time.time()-started])
        eval_limit = 1 if args.smoke_test else None
        small=evaluate(model,ae,small_loader,cfg,device,latent_coords,latent_patch,latent_full,eval_limit); full=evaluate(model,ae,val_loader,cfg,device,latent_coords,latent_patch,latent_full,eval_limit)
        append_csv(val_csv,[epoch,*[full[k] for k in ['image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae']]])
        checkpoint(out/'checkpoints/last.pt',epoch,model,opt,sched,best,cfg,{'small':small,'full':full})
        if full['ssim'] > best: best=full['ssim']; checkpoint(out/'checkpoints/best_ssim.pt',epoch,model,opt,sched,best,cfg,{'small':small,'full':full})
        print(json.dumps({'epoch':epoch,'train_image_l1':avg,'small':small,'full':full,'peak_gpu_mem_mb':peak,'best_ssim':best}),flush=True)


if __name__ == '__main__': main()
