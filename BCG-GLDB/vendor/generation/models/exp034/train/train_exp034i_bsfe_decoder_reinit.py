"""Exp034I: load Exp034G model, train BSFE, and use a freshly initialized PET decoder."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
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
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--init-ckpt',required=True); p.add_argument('--smoke-test',action='store_true'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); torch.manual_seed(int(cfg['experiment'].get('seed',42))); device=torch.device(cfg['training'].get('device','cuda'))
    out=ROOT/cfg['experiment']['output_dir']; (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    image_size=tuple(cfg['data']['image_size']); patch_size=tuple(cfg['data']['patch_size']); latent_full=tuple(v//4 for v in image_size); latent_patch=tuple(v//4 for v in patch_size); coords=build_patch_grid(latent_full,latent_patch)
    train_ds=build_dataset(cfg,'train'); nw=int(cfg['training'].get('num_workers',0)); common=dict(num_workers=nw,pin_memory=bool(cfg['training'].get('pin_memory',False)),persistent_workers=nw>0)
    train=DataLoader(train_ds,batch_size=int(cfg['training']['batch_size']),shuffle=True,drop_last=True,**common); sets=[build_dataset(cfg,s,train_dataset=train_ds) for s in cfg['evaluation']['val_splits']]; val_ds=sets[0] if len(sets)==1 else ConcatDataset(sets); val=DataLoader(val_ds,batch_size=1,shuffle=False,**common); small=DataLoader(Subset(val_ds,range(min(len(val_ds),int(cfg['evaluation']['small_val_subjects'])))),batch_size=1)
    ae=load_autoencoder(cfg['ae']['kind'],ROOT/cfg['ae']['checkpoint_path'],device)
    # Keep the loaded encoder/latent interface, but discard all checkpoint decoder weights.
    ae.decoder=PatchBackboneDecoder(ae.image_channels,ae.latent_channels,ae.base_channels,ae.norm_groups,ae.dropout).to(device)
    for q in ae.encoder_backbone.parameters(): q.requires_grad_(False)
    for q in ae.parameters(): q.requires_grad_(False)
    for q in ae.decoder.parameters(): q.requires_grad_(True)
    bsfe,head=build_bsfe_models(cfg,latent_full,device); state=torch.load(ROOT/cfg['bsfe']['checkpoint_path'],map_location=device,weights_only=False); bsfe.load_state_dict(state['bsfe']); head.load_state_dict(state['head']); del state
    for q in head.parameters(): q.requires_grad_(False)
    for q in bsfe.parameters(): q.requires_grad_(True)
    ids,_=infer_feature_type_ids(train_ds.table_cols); m,tb=cfg['model'],cfg['table']; model=BSFETableMambaUNet3D(table_num_features=len(train_ds.table_cols),feature_type_ids=ids,in_channels=4,out_channels=4,channels=tuple(m['channels']),bsfe_channels=tuple(m['bsfe_channels']),time_dim=int(m['time_dim']),table_token_dim=int(tb['token_dim']),table_heads=int(tb['heads']),table_token_grids=tb['token_grids'],norm_groups=int(m['norm_groups']),dropout=0.0,table_lambda_init=float(tb['lambda_init']),latent_shape=tuple(m['latent_shape']),mamba_token_dim=int(m['mamba_token_dim']),mamba_depth=int(m['mamba_depth']),mamba_d_state=int(m['mamba_d_state']),mamba_d_conv=int(m['mamba_d_conv']),mamba_expand=int(m['mamba_expand']),mamba_line_chunk_size=int(m['mamba_line_chunk_size'])).to(device)
    src=torch.load(a.init_ckpt,map_location='cpu',weights_only=False)['model_state_dict']; tgt=model.state_dict(); hit={k:v for k,v in src.items() if k in tgt and tgt[k].shape==v.shape}; model.load_state_dict(hit,strict=False); loaded=len(hit); del src,hit
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':float(cfg['training']['learning_rate'])},{'params':bsfe.parameters(),'lr':float(cfg['training']['bsfe_learning_rate'])},{'params':ae.decoder.parameters(),'lr':float(cfg['training']['decoder_learning_rate'])}],weight_decay=float(cfg['training']['weight_decay']))
    epochs=1 if a.smoke_test else int(cfg['training']['epochs']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(epochs,1)); amp=cfg['training'].get('amp','bf16'); dtype=torch.bfloat16 if amp=='bf16' else torch.float16; scaler=torch.cuda.amp.GradScaler(enabled=amp=='fp16')
    ensure_csv(out/'logs/train_log.csv',['epoch','image_l1','lr_model','lr_bsfe','lr_decoder','peak_gpu_mem_mb','epoch_time_sec']); ensure_csv(out/'logs/val_log.csv',['epoch','image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae']); print(json.dumps({'experiment':cfg['experiment']['name'],'loaded_exp034g_tensors':loaded,'decoder_initialized':'fresh','bsfe_trainable':True,'decoder_parameters':sum(q.numel() for q in ae.decoder.parameters())}),flush=True)
    def run(b):
        pet=b['pet'].to(device).float(); mri=b['mri'].to(device).float(); table=b['table'].to(device).float(); mask=b['table_mask'].to(device).float(); t=torch.ones(pet.shape[0],device=device,dtype=torch.long); pyramid=bsfe(mri)['pyramid']; z=model(mri_endpoint(mri,4),t,pyramid,table,mask); recon=decode(ae,z,pet.shape,coords,latent_patch,latent_full); return F.l1_loss(recon,pet),recon,pet
    @torch.no_grad()
    def ev(loader,limit=None):
        model.eval(); ae.decoder.eval(); rows=[]
        for i,b in enumerate(loader):
            loss,r,pet=run(b); mm=compute_image_metrics_detailed(r,pet,mask=b['brain_mask'].to(device).float(),data_range=2.0); rows.append({'image_l1':float(loss),**{k:float(mm[k]) for k in ('ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')}})
            if limit and i+1>=limit: break
        return {k:float(np.mean([x[k] for x in rows])) for k in rows[0]}
    accum=max(int(cfg['training'].get('grad_accum',1)),1); best=-float('inf')
    for epoch in range(1,epochs+1):
        model.train(); bsfe.train(); ae.decoder.train(); opt.zero_grad(set_to_none=True); losses=[]; start=time.time(); torch.cuda.reset_peak_memory_stats(device)
        for i,b in enumerate(train):
            with torch.autocast('cuda',dtype=dtype,enabled=amp!='off'): loss,_,_=run(b)
            scaler.scale(loss/accum).backward()
            if (i+1)%accum==0 or i+1==len(train): scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if a.smoke_test: break
        sched.step(); avg=float(np.mean(losses)); peak=torch.cuda.max_memory_allocated(device)/1024**2; full=ev(val,1 if a.smoke_test else None); append_csv(out/'logs/train_log.csv',[epoch,avg,opt.param_groups[0]['lr'],opt.param_groups[1]['lr'],opt.param_groups[2]['lr'],peak,time.time()-start]); append_csv(out/'logs/val_log.csv',[epoch,*[full[k] for k in ('image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')]])
        ck={'epoch':epoch,'model_state_dict':model.state_dict(),'bsfe_state_dict':bsfe.state_dict(),'decoder_state_dict':ae.decoder.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sched.state_dict(),'best_ssim':best,'metrics':{'full':full},'config':cfg}; torch.save(ck,out/'checkpoints/last.pt')
        if full['ssim']>best: best=full['ssim']; ck['best_ssim']=best; torch.save(ck,out/'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch':epoch,'train_image_l1':avg,'full':full,'peak_gpu_mem_mb':peak,'best_ssim':best}),flush=True)
if __name__=='__main__': main()
