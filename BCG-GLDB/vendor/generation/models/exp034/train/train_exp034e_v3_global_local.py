"""Train Exp034E_v3 gated global-local Mamba fusion."""
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
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034e_v3_global_local_unet3d import GlobalLocalMambaUNet3D
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv

def grad_l1(a,b): return sum(F.l1_loss(a.diff(dim=d),b.diff(dim=d)) for d in (2,3,4))/3
def low_l1(a,b): return F.l1_loss(F.avg_pool3d(a,5,2,2),F.avg_pool3d(b,5,2,2))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--init-ckpt',required=True); p.add_argument('--resume'); p.add_argument('--smoke-test',action='store_true'); a=p.parse_args(); cfg=yaml.safe_load(open(a.config)); torch.manual_seed(int(cfg['experiment']['seed'])); dev=torch.device(cfg['training']['device']); out=ROOT/cfg['experiment']['output_dir']; (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'logs').mkdir(exist_ok=True)
 im,pa=tuple(cfg['data']['image_size']),tuple(cfg['data']['patch_size']); lf,lp=tuple(x//4 for x in im),tuple(x//4 for x in pa); coords=build_patch_grid(lf,lp); ds=build_dataset(cfg,'train'); nw=int(cfg['training']['num_workers']); kw=dict(num_workers=nw,pin_memory=True,persistent_workers=True); tr=DataLoader(ds,batch_size=1,shuffle=True,drop_last=True,**kw); sets=[build_dataset(cfg,s,train_dataset=ds) for s in cfg['evaluation']['val_splits']]; vd=ConcatDataset(sets); va=DataLoader(vd,batch_size=1,shuffle=False,**kw)
 ae=load_autoencoder(cfg['ae']['kind'],ROOT/cfg['ae']['checkpoint_path'],dev); mae=load_autoencoder(cfg['mri_ae']['kind'],ROOT/cfg['mri_ae']['checkpoint_path'],dev)
 for q in list(ae.parameters())+list(mae.parameters()): q.requires_grad_(False)
 m=cfg['model']; model=GlobalLocalMambaUNet3D(in_channels=4,out_channels=4,channels=tuple(m['channels']),time_dim=int(m['time_dim']),norm_groups=int(m['norm_groups']),dropout=float(m['dropout']),mamba_token_dim=int(m['mamba_token_dim']),mamba_depth=int(m['mamba_depth']),mamba_d_state=int(m['mamba_d_state']),mamba_d_conv=int(m['mamba_d_conv']),mamba_expand=int(m['mamba_expand']),mamba_line_chunk_size=int(m['mamba_line_chunk_size'])).to(dev)
 src=torch.load(ROOT/a.init_ckpt,map_location='cpu',weights_only=False)['model_state_dict']; target=model.state_dict(); hit={k:v for k,v in src.items() if k in target and target[k].shape==v.shape}; model.load_state_dict(hit,strict=False); del src,hit
 base,glob=[],[]
 for n,q in model.named_parameters(): (glob if n.startswith(('global_context','fuse_','context_to_')) else base).append(q)
 opt=torch.optim.AdamW([{'params':base,'lr':float(cfg['training']['learning_rate'])},{'params':glob,'lr':float(cfg['training']['global_learning_rate'])}],weight_decay=float(cfg['training']['weight_decay'])); epochs=1 if a.smoke_test else int(cfg['training']['epochs']); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(epochs,1)); amp=cfg['training']['amp']; dt=torch.bfloat16 if amp=='bf16' else torch.float16; sc=torch.cuda.amp.GradScaler(enabled=amp=='fp16'); ensure_csv(out/'logs/train_log.csv',['epoch','loss_total','image_l1','global_l1','local_gradient_l1','peak_gpu_mem_mb']); ensure_csv(out/'logs/val_log.csv',['epoch','image_l1','global_l1','local_gradient_l1','ssim','psnr'])
 start_epoch=1; best=-1.
 if a.resume:
  state=torch.load(ROOT/a.resume,map_location=dev,weights_only=False); model.load_state_dict(state['model_state_dict']); opt.load_state_dict(state['optimizer_state_dict']); sch.load_state_dict(state['scheduler_state_dict']); start_epoch=int(state['epoch'])+1; best=float(state.get('best_ssim',-1.)); del state
 print(json.dumps({'experiment':cfg['experiment']['name'],'loaded_backbone_tensors':len(target)-len([k for k in target if k not in model.state_dict()]),'mri_ae_checkpoint':cfg['mri_ae']['checkpoint_path'],'global_local_fusion':'gated_bottleneck_to_dec2_dec1','resume':a.resume,'start_epoch':start_epoch}),flush=True)
 def run(b):
  pet=b['pet'].to(dev).float(); mri=b['mri'].to(dev).float()
  with torch.no_grad(): x=mae.encode(mri)
  z=model(x,torch.ones(pet.shape[0],device=dev,dtype=torch.long)); rec=decode(ae,z,pet.shape,coords,lp,lf); image=F.l1_loss(rec,pet); glob=low_l1(rec,pet); loc=grad_l1(rec,pet); w=cfg['loss']; return w['image_l1_weight']*image+w['global_l1_weight']*glob+w['local_gradient_l1_weight']*loc,{'image_l1':image,'global_l1':glob,'local_gradient_l1':loc},rec,pet
 @torch.no_grad()
 def ev(limit=None):
  model.eval(); rows=[]
  for i,b in enumerate(va):
   _,ts,r,p=run(b); mm=compute_image_metrics_detailed(r,p,mask=b['brain_mask'].to(dev).float(),data_range=2.0); rows.append({**{k:float(v) for k,v in ts.items()},'ssim':float(mm['ssim']),'psnr':float(mm['psnr'])})
   if limit and i+1>=limit: break
  return {k:float(np.mean([x[k] for x in rows])) for k in rows[0]}
 accum=int(cfg['training']['grad_accum'])
 for e in range(start_epoch,epochs+1):
  model.train(); opt.zero_grad(set_to_none=True); ls=[]; terms=[]; torch.cuda.reset_peak_memory_stats(dev)
  for i,b in enumerate(tr):
   with torch.autocast('cuda',dtype=dt,enabled=amp!='off'): loss,t,_,_=run(b)
   sc.scale(loss/accum).backward()
   if (i+1)%accum==0 or i+1==len(tr): sc.step(opt);sc.update();opt.zero_grad(set_to_none=True)
   ls.append(float(loss.detach()));terms.append({k:float(v.detach()) for k,v in t.items()})
   if a.smoke_test: break
  sch.step(); mt={k:float(np.mean([x[k] for x in terms])) for k in terms[0]}; full=ev(1 if a.smoke_test else None); peak=torch.cuda.max_memory_allocated(dev)/1024**2; append_csv(out/'logs/train_log.csv',[e,float(np.mean(ls)),mt['image_l1'],mt['global_l1'],mt['local_gradient_l1'],peak]);append_csv(out/'logs/val_log.csv',[e,*[full[k] for k in ('image_l1','global_l1','local_gradient_l1','ssim','psnr')]]); state={'epoch':e,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sch.state_dict(),'best_ssim':best,'metrics':{'full':full},'config':cfg};torch.save(state,out/'checkpoints/last.pt');
  if full['ssim']>best: best=full['ssim'];state['best_ssim']=best;torch.save(state,out/'checkpoints/best_ssim.pt')
  print(json.dumps({'epoch':e,'train_loss':float(np.mean(ls)),'loss_terms':mt,'full':full,'peak_gpu_mem_mb':peak,'best_ssim':best}),flush=True)
if __name__=='__main__':main()
