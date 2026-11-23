"""Exp034M staged curriculum: A_v3 backbone, then table, BSFE, Mamba, decoder and ROI losses."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp026.models.table_interaction import MissingAwareTableTokenizer
from exp026.utils.patch_codec import build_patch_grid
from exp032e.utils import load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp034m_curriculum_unet3d import Exp034MCurriculumUNet3D
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_bsfe_table_jointprior_latent_bridge_t1_to_fdg import infer_feature_type_ids
from train.train_exp032j_sampling_aligned import build_dataset
from train.train_exp034a_mrireshape_onestep_bridge import append_csv, decode, ensure_csv

class TablePredictor(nn.Module):
    def __init__(self, n):
        super().__init__(); self.net=nn.Sequential(nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.LayerNorm(4),nn.Linear(4,32),nn.SiLU(),nn.Linear(32,n))
    def forward(self,x): return self.net(x)

def roi_loss(pred,target,labels,mask):
    p_losses=[]; c_losses=[]
    for b in range(pred.shape[0]):
        lab=labels[b,0].long(); ok=(lab>0)&(lab<=98)&(mask[b,0]>.5); lab=lab[ok]; p=pred[b,0][ok]; y=target[b,0][ok]
        cnt=torch.bincount(lab,minlength=99)[1:].float(); present=cnt>=32
        if present.any():
            ids=torch.where(present)[0]+1; pm=torch.stack([p[lab==i].mean() for i in ids]).clamp(-1,1); ym=torch.stack([y[lab==i].mean() for i in ids]).clamp(-1,1); scale=y.abs().mean().detach().clamp_min(.25); w=cnt[present].sqrt(); w=w/w.sum()
            p_losses.append((F.smooth_l1_loss(pm/scale,ym/scale,reduction='none')*w).sum()); c_losses.append((F.smooth_l1_loss((pm-pm.mean())/scale,(ym-ym.mean())/scale,reduction='none')*w).sum())
    z=pred.sum()*0; return (torch.stack(p_losses).mean() if p_losses else z, torch.stack(c_losses).mean() if c_losses else z)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--smoke-stage',type=int); ap.add_argument('--all-active', action='store_true'); ap.add_argument('--resume', action='store_true'); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config))
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0')); local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if distributed:
        dist.init_process_group('nccl'); torch.cuda.set_device(local_rank)
    torch.manual_seed(int(cfg['experiment']['seed']) + rank); device=torch.device('cuda', local_rank)
    is_main = rank == 0
    out=ROOT/cfg['experiment']['output_dir']
    if is_main: (out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    if distributed: dist.barrier()
    im,pa=tuple(cfg['data']['image_size']),tuple(cfg['data']['patch_size']); full,patch=tuple(x//4 for x in im),tuple(x//4 for x in pa); coords=build_patch_grid(full,patch)
    train_ds=build_dataset(cfg,'train'); val_ds=ConcatDataset([build_dataset(cfg,s,train_dataset=train_ds) for s in cfg['evaluation']['val_splits']]); ids,_=infer_feature_type_ids(train_ds.table_cols)
    pet_ae=load_autoencoder(cfg['ae']['kind'],ROOT/cfg['ae']['checkpoint_path'],device); mri_ae=load_autoencoder(cfg['mri_ae']['kind'],ROOT/cfg['mri_ae']['checkpoint_path'],device)
    for q in list(pet_ae.parameters())+list(mri_ae.parameters()): q.requires_grad_(False)
    m=cfg['model']; model=Exp034MCurriculumUNet3D(len(train_ds.table_cols),ids,tuple(m['channels']),int(m['time_dim']),int(m['norm_groups']),int(m['attention_heads']),int(cfg['table']['token_dim'])).to(device)
    model.tokenizer=MissingAwareTableTokenizer(len(train_ds.table_cols),int(cfg['table']['token_dim']),ids,num_feature_types=5).to(device)
    table_head=TablePredictor(len(train_ds.table_cols)).to(device); bsfe,bsfe_head=build_bsfe_models(cfg,full,device)
    bsfe_ckpt = ROOT / cfg['bsfe']['checkpoint_path']
    bsfe_state = torch.load(bsfe_ckpt, map_location=device)
    bsfe.load_state_dict(bsfe_state['bsfe'], strict=True)
    bsfe_head.load_state_dict(bsfe_state['head'], strict=True)
    if is_main:
        print(json.dumps({'bsfe_checkpoint': str(bsfe_ckpt), 'bsfe_loaded': True,
                          'all_new_modules_random_init': True, 'mri_ae_frozen': True,
                          'pet_ae_frozen_until_stage5': True, 'world_size': int(os.environ.get('WORLD_SIZE', '1'))}), flush=True)
        ensure_csv(out/'logs/metrics.csv',['epoch','stage','batch_size','loss','image_l1','table_aux','bsfe_guide','global_l1','local_grad','roi_profile','roi_contrast','ssim','psnr','peak_mb','seconds'])
    def run(batch,stage,network=None):
        pet=batch['pet'].to(device).float(); mri=batch['mri'].to(device).float(); table=batch['table'].to(device).float(); table_mask=batch['table_mask'].to(device).float()
        with torch.no_grad(): x1=mri_ae.encode(mri)
        pyramid=None; bs_pet=None
        if stage>=3:
            features=bsfe(mri); pyramid=features['pyramid']; bs_pet=bsfe_head(pyramid,features['global'],features['tokens'])['pet_prediction']
        network = model if network is None else network
        z=network(x1,torch.ones(pet.shape[0],device=device,dtype=torch.long),table,table_mask,pyramid,stage); recon=decode(pet_ae,z,pet.shape,coords,patch,full)
        image=F.l1_loss(recon,pet); tab=((F.smooth_l1_loss(table_head(z),table,reduction='none')*(table_mask<.5)).sum()/(table_mask<.5).sum().clamp_min(1)) if stage>=2 else image*0
        guide=F.l1_loss(recon,bs_pet) if stage>=3 else image*0; global_l1=F.l1_loss(F.avg_pool3d(recon,5,2,2),F.avg_pool3d(pet,5,2,2)) if stage>=4 else image*0; local=sum(F.l1_loss(recon.diff(d),pet.diff(d)) for d in (2,3,4))/3 if stage>=4 else image*0
        rp,rc=roi_loss(recon,pet,batch['parc98'].to(device),batch['brain_mask'].to(device).float()) if stage>=6 else (image*0,image*0)
        ssim=1-compute_image_metrics_detailed(recon,pet,data_range=2.0)['ssim'] if stage>=6 else image*0
        total=image+.05*tab+.02*guide+(.1*global_l1+.05*local if stage>=4 else 0)+(.2*ssim+.03*rp+.02*rc if stage>=6 else 0)
        return total,(image,tab,guide,global_l1,local,rp,rc),recon,pet
    raw_model = model
    if distributed: model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    def params(stage):
        outp=list(raw_model.parameters())
        if stage>=2: outp+=list(table_head.parameters())
        if stage>=3: outp+=list(bsfe.parameters())+list(bsfe_head.parameters())
        if stage>=5: outp+=list(pet_ae.parameters())
        return outp
    def sync_aux_grads(stage):
        if not distributed: return
        modules = ([] if stage < 2 else [table_head]) + ([] if stage < 3 else [bsfe, bsfe_head]) + ([] if stage < 5 else [pet_ae])
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(dist.get_world_size())
    def evaluate(stage):
        raw_model.eval(); table_head.eval(); bsfe.eval(); bsfe_head.eval(); batch=next(iter(DataLoader(val_ds,batch_size=1,shuffle=False)))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): _,_,recon,pet=run(batch,stage,raw_model)
        met=compute_image_metrics_detailed(recon,pet,mask=batch['brain_mask'].to(device).float(),data_range=2.0); return float(met['ssim']),float(met['psnr'])
    final_epoch=1 if args.smoke_stage else int(cfg['training'].get('epochs', 300)); begin=(args.smoke_stage-1)*50+1 if args.smoke_stage else 1; current=None; opt=None
    resume_ckpt = out/'checkpoints/last.pt'
    if args.resume and resume_ckpt.exists():
        state=torch.load(resume_ckpt,map_location='cpu'); raw_model.load_state_dict(state['model'],strict=True); table_head.load_state_dict(state['table_head'],strict=True); bsfe.load_state_dict(state['bsfe'],strict=True); bsfe_head.load_state_dict(state['bsfe_head'],strict=True); pet_ae.load_state_dict(state['pet_ae'],strict=True); begin=int(state['epoch'])+1
        if is_main: print(json.dumps({'resume_loaded': True, 'resume_epoch': begin-1}), flush=True)
    for epoch in range(begin,begin+final_epoch):
        stage=args.smoke_stage or (6 if args.all_active else min(6,(epoch-1)//50+1))
        if stage!=current:
            current=stage
            for q in pet_ae.parameters(): q.requires_grad_(stage>=5)
            opt=torch.optim.AdamW(params(stage),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
        bs=int(cfg['training']['stage_batch_sizes'][stage-1]); sampler=DistributedSampler(train_ds, num_replicas=int(os.environ.get('WORLD_SIZE','1')), rank=rank, shuffle=True, drop_last=True) if distributed else None; loader=DataLoader(train_ds,batch_size=bs,shuffle=sampler is None, sampler=sampler,drop_last=True,num_workers=int(cfg['training']['num_workers']),pin_memory=True,persistent_workers=True)
        if sampler: sampler.set_epoch(epoch)
        model.train(); table_head.train(stage>=2); bsfe.train(stage>=3); bsfe_head.train(stage>=3); pet_ae.train(stage>=5); opt.zero_grad(set_to_none=True); rows=[]; t0=time.time(); torch.cuda.reset_peak_memory_stats(device)
        for i,batch in enumerate(loader):
            with torch.autocast('cuda',dtype=torch.bfloat16): loss,terms,_,_=run(batch,stage)
            loss.backward(); sync_aux_grads(stage); opt.step(); opt.zero_grad(set_to_none=True); rows.append([float(x.detach()) for x in terms])
            if args.smoke_stage: break
        if distributed: dist.barrier()
        if is_main:
            ssim,psnr=evaluate(stage); terms=np.mean(rows,axis=0); peak=torch.cuda.max_memory_allocated(device)/1024**2
            append_csv(out/'logs/metrics.csv',[epoch,stage,bs,float(loss.detach()),*terms,ssim,psnr,peak,time.time()-t0])
            torch.save({'epoch':epoch,'stage':stage,'model':raw_model.state_dict(),'table_head':table_head.state_dict(),'bsfe':bsfe.state_dict(),'bsfe_head':bsfe_head.state_dict(),'pet_ae':pet_ae.state_dict(),'config':cfg},out/'checkpoints/last.pt')
            print(json.dumps({'epoch':epoch,'stage':stage,'batch_size_per_gpu':bs,'global_batch_size':bs*int(os.environ.get('WORLD_SIZE','1')),'ssim':ssim,'psnr':psnr,'peak_gpu_mem_mb':peak}),flush=True)
        if distributed: dist.barrier()
    if distributed: dist.destroy_process_group()
if __name__=='__main__': main()
