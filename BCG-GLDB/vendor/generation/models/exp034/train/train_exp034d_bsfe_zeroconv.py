from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader, ConcatDataset, Subset
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from exp026.utils.patch_codec import build_patch_grid
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034d_bsfe_zeroconv_unet3d import BSFEZeroConvUNet3D
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import decode, ensure_csv, append_csv, mri_endpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--init-ckpt',required=True); p.add_argument('--smoke-test',action='store_true'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); device=torch.device(cfg['training'].get('device','cuda')); out=REPO_ROOT/cfg['experiment']['output_dir']; (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    image_size=tuple(cfg['data']['image_size']); patch_size=tuple(cfg['data']['patch_size']); latent_full=tuple(v//4 for v in image_size); latent_patch=tuple(v//4 for v in patch_size); latent_coords=build_patch_grid(latent_full,latent_patch)
    train_ds=build_dataset(cfg,'train'); nw=int(cfg['training'].get('num_workers',0)); common=dict(num_workers=nw,pin_memory=bool(cfg['training'].get('pin_memory',False)),persistent_workers=nw>0); train=DataLoader(train_ds,batch_size=int(cfg['training']['batch_size']),shuffle=True,drop_last=True,**common); sets=[build_dataset(cfg,s,train_dataset=train_ds) for s in cfg['evaluation']['val_splits']]; vd=sets[0] if len(sets)==1 else ConcatDataset(sets); val=DataLoader(vd,batch_size=int(cfg['training']['batch_size']),shuffle=False,**common); small=DataLoader(Subset(vd,range(min(len(vd),int(cfg['evaluation'].get('small_val_subjects',4))))),batch_size=int(cfg['training']['batch_size']),shuffle=False)
    ae=load_autoencoder(cfg['ae']['kind'],REPO_ROOT/cfg['ae']['checkpoint_path'],device)
    bsfe,head=build_bsfe_models(cfg,latent_full,device); ck=torch.load(REPO_ROOT/cfg['bsfe']['checkpoint_path'],map_location=device); bsfe.load_state_dict(ck['bsfe']); head.load_state_dict(ck['head']); del ck,head
    for p in bsfe.parameters(): p.requires_grad_(False)
    m=cfg['model']; model=BSFEZeroConvUNet3D(channels=tuple(m['channels']),bsfe_channels=tuple(m['bsfe_channels']),time_dim=int(m['time_dim']),norm_groups=int(m['norm_groups']),dropout=float(m.get('dropout',0))).to(device)
    init=torch.load(a.init_ckpt,map_location=device); model.load_state_dict({k:v for k,v in init['model_state_dict'].items() if k in model.state_dict() and model.state_dict()[k].shape==v.shape},strict=False); del init
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training'].get('weight_decay',1e-4))); epochs=1 if a.smoke_test else int(cfg['training']['epochs']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(epochs,1)); amp=cfg['training'].get('amp','bf16'); dtype=torch.bfloat16 if amp=='bf16' else torch.float16
    ensure_csv(out/'logs/train_log.csv',['epoch','image_l1','lr','peak_gpu_mem_mb','epoch_time_sec']); ensure_csv(out/'logs/val_log.csv',['epoch','image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae']); accum=max(int(cfg['training'].get('grad_accum',2)),1); best=-float('inf'); scaler=torch.cuda.amp.GradScaler(enabled=amp=='fp16')
    def run(batch):
        pet=batch['pet'].to(device).float(); mri=batch['mri'].to(device).float(); t=torch.ones(pet.shape[0],device=device,dtype=torch.long)
        with torch.no_grad(): pyramid=bsfe(mri)['pyramid']
        z=model(mri_endpoint(mri,4),t,pyramid); recon=decode(ae,z,pet.shape,latent_coords,latent_patch,latent_full); return F.l1_loss(recon,pet),recon,pet
    @torch.no_grad()
    def ev(loader,limit=None):
        model.eval(); rows=[]
        for i,b in enumerate(loader):
            loss,r,pet=run(b); mm=compute_image_metrics_detailed(r,pet,mask=b['brain_mask'].to(device).float(),data_range=2.0); rows.append({'image_l1':float(loss),**{k:float(mm[k]) for k in ('ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')}})
            if limit and i+1>=limit: break
        return {k:float(np.mean([x[k] for x in rows])) for k in rows[0]}
    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); ls=[]; st=time.time(); torch.cuda.reset_peak_memory_stats(device)
        for i,b in enumerate(train):
            with torch.autocast('cuda',dtype=dtype,enabled=amp!='off'): loss,_,_=run(b); scaler.scale(loss/accum).backward()
            if (i+1)%accum==0 or i+1==len(train): scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            ls.append(float(loss.detach()));
            if a.smoke_test: break
        sched.step(); sm=ev(small,1 if a.smoke_test else None); full=ev(val,1 if a.smoke_test else None); avg=float(np.mean(ls)); peak=torch.cuda.max_memory_allocated(device)/1024**2; append_csv(out/'logs/train_log.csv',[epoch,avg,opt.param_groups[0]['lr'],peak,time.time()-st]); append_csv(out/'logs/val_log.csv',[epoch,*[full[k] for k in ('image_l1','ssim','psnr','mae','rmse','brain_ssim','brain_psnr','brain_mae')]])
        state={'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sched.state_dict(),'best_ssim':best,'metrics':{'small':sm,'full':full}}; torch.save(state,out/'checkpoints/last.pt')
        if full['ssim']>best: best=full['ssim']; state['best_ssim']=best; torch.save(state,out/'checkpoints/best_ssim.pt')
        print(json.dumps({'epoch':epoch,'train_image_l1':avg,'full':full,'peak_gpu_mem_mb':peak,'best_ssim':best}),flush=True)
if __name__=='__main__': main()
