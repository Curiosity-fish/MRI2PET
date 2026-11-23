"""Exp034D_v2: frozen MRI-AE endpoint and frozen BSFE PET structural guide."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from exp026.utils.patch_codec import build_patch_grid
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034d_bsfe_zeroconv_unet3d import BSFEZeroConvUNet3D
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--init-ckpt', required=True); p.add_argument('--smoke-test', action='store_true'); a = p.parse_args()
    cfg = yaml.safe_load(open(a.config)); torch.manual_seed(int(cfg['experiment'].get('seed', 42))); device = torch.device(cfg['training'].get('device', 'cuda'))
    out = ROOT / cfg['experiment']['output_dir']; (out/'checkpoints').mkdir(parents=True, exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    image_size, patch_size = tuple(cfg['data']['image_size']), tuple(cfg['data']['patch_size']); latent_full, latent_patch = tuple(v//4 for v in image_size), tuple(v//4 for v in patch_size); coords = build_patch_grid(latent_full, latent_patch)
    train_ds = build_dataset(cfg, 'train'); nw = int(cfg['training'].get('num_workers', 0)); common = dict(num_workers=nw, pin_memory=bool(cfg['training'].get('pin_memory', False)), persistent_workers=nw > 0)
    train = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, drop_last=True, **common); sets = [build_dataset(cfg, s, train_dataset=train_ds) for s in cfg['evaluation']['val_splits']]; vd = sets[0] if len(sets)==1 else ConcatDataset(sets); val = DataLoader(vd, batch_size=1, shuffle=False, **common); small = DataLoader(Subset(vd, range(min(len(vd), int(cfg['evaluation']['small_val_subjects'])))), batch_size=1)
    pet_ae = load_autoencoder(cfg['ae']['kind'], ROOT/cfg['ae']['checkpoint_path'], device); mri_ae = load_autoencoder(cfg['mri_ae']['kind'], ROOT/cfg['mri_ae']['checkpoint_path'], device)
    for module in (pet_ae, mri_ae):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    bsfe, head = build_bsfe_models(cfg, latent_full, device); state = torch.load(ROOT/cfg['bsfe']['checkpoint_path'], map_location=device, weights_only=False); bsfe.load_state_dict(state['bsfe']); head.load_state_dict(state['head']); del state
    for module in (bsfe, head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    m = cfg['model']; model = BSFEZeroConvUNet3D(in_channels=4, out_channels=4, channels=tuple(m['channels']), bsfe_channels=tuple(m['bsfe_channels']), time_dim=int(m['time_dim']), norm_groups=int(m['norm_groups']), dropout=float(m.get('dropout', 0))).to(device)
    init = torch.load(ROOT/a.init_ckpt, map_location='cpu', weights_only=False); target = model.state_dict(); compatible = {k:v for k,v in init['model_state_dict'].items() if k in target and target[k].shape == v.shape}; model.load_state_dict(compatible, strict=False); loaded = len(compatible); del init, compatible
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training'].get('weight_decay', 1e-4))); epochs = 1 if a.smoke_test else int(cfg['training']['epochs']); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1)); amp = cfg['training'].get('amp', 'bf16'); dtype = torch.bfloat16 if amp == 'bf16' else torch.float16; scaler = torch.cuda.amp.GradScaler(enabled=amp == 'fp16')
    ensure_csv(out/'logs/train_log.csv', ['epoch','loss_total','loss_image_l1','loss_bsfe_guide_l1','lr','peak_gpu_mem_mb','epoch_time_sec']); ensure_csv(out/'logs/val_log.csv', ['epoch','image_l1','bsfe_guide_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae'])
    print(json.dumps({'experiment': cfg['experiment']['name'], 'loaded_d_tensors': loaded, 'mri_ae_frozen': True, 'bsfe_frozen': True, 'bsfe_guide_weight': cfg['loss']['bsfe_guide_l1_weight']}), flush=True)
    def run(batch):
        pet, mri = batch['pet'].to(device).float(), batch['mri'].to(device).float(); t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        with torch.no_grad():
            x1 = mri_ae.encode(mri); features = bsfe(mri); pyramid = features['pyramid']; bsfe_pet = head(features['pyramid'], features['global'], features['tokens'])['pet_prediction']
        if tuple(x1.shape[1:]) != (4, *latent_full): raise RuntimeError(f'Unexpected MRI AE latent {tuple(x1.shape)}')
        z = model(x1, t, pyramid); recon = decode(pet_ae, z, pet.shape, coords, latent_patch, latent_full); image_l1 = F.l1_loss(recon, pet); guide = F.l1_loss(recon, bsfe_pet)
        loss = float(cfg['loss']['image_l1_weight'])*image_l1 + float(cfg['loss']['bsfe_guide_l1_weight'])*guide
        return loss, {'image_l1': image_l1, 'bsfe_guide_l1': guide}, recon, pet
    @torch.no_grad()
    def ev(loader, limit=None):
        model.eval(); rows=[]
        for i,b in enumerate(loader):
            _, terms, recon, pet = run(b); mm=compute_image_metrics_detailed(recon, pet, mask=b['brain_mask'].to(device).float(), data_range=2.0); rows.append({'image_l1':float(terms['image_l1']), 'bsfe_guide_l1':float(terms['bsfe_guide_l1']), **{k:float(mm[k]) for k in ('ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')}})
            if limit and i+1 >= limit: break
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    accum = max(int(cfg['training'].get('grad_accum', 1)), 1); best = -float('inf')
    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); losses=[]; terms_list=[]; start=time.time(); torch.cuda.reset_peak_memory_stats(device)
        for i,b in enumerate(train):
            with torch.autocast('cuda', dtype=dtype, enabled=amp != 'off'): loss, terms, _, _ = run(b)
            scaler.scale(loss/accum).backward()
            if (i+1)%accum == 0 or i+1 == len(train): scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach())); terms_list.append({k:float(v.detach()) for k,v in terms.items()})
            if a.smoke_test: break
        sched.step(); mean_terms={k:float(np.mean([x[k] for x in terms_list])) for k in terms_list[0]}; full=ev(val, 1 if a.smoke_test else None); peak=torch.cuda.max_memory_allocated(device)/1024**2
        append_csv(out/'logs/train_log.csv', [epoch,float(np.mean(losses)),mean_terms['image_l1'],mean_terms['bsfe_guide_l1'],opt.param_groups[0]['lr'],peak,time.time()-start]); append_csv(out/'logs/val_log.csv', [epoch,*[full[k] for k in ('image_l1','bsfe_guide_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')]])
        ck={'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sched.state_dict(),'best_ssim':best,'metrics':{'full':full},'config':cfg}; torch.save(ck,out/'checkpoints/last.pt')
        if full['ssim'] > best: best=full['ssim']; ck['best_ssim']=best; torch.save(ck,out/'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch':epoch,'train_loss':float(np.mean(losses)),'loss_terms':mean_terms,'full':full,'peak_gpu_mem_mb':peak,'best_ssim':best}), flush=True)
if __name__ == '__main__': main()
