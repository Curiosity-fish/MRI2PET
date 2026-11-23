from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PKGROOT=ROOT/'MRI2PET'
if str(PKGROOT) not in sys.path: sys.path.insert(0,str(PKGROOT))
from MRI2PET.classier.medicalnet.resnet import resnet50
from MRI2PET.classier.paired_t1_dx_dataset import _load_volume, _resize_volume
from MRI2PET.models.exp037c_global_local_vit_unet3d import GlobalViT3D
from MRI2PET.exp.exp037B_v4.exp037B_v4_unet3d import ClinicalTableEncoder

TABLE_COLS=['PTDEMOG_PTGENDER','PTDEMOG_PTEDUCAT','AGE_derived','APOE4_COUNT','APOE4_CARRIER','FAMILY_HISTORY_AD','PARENT_HISTORY_AD','SIBLING_HISTORY_AD','MMSE_MMSCORE','ADAS_TOTAL13','MOCA_MOCA','MEDHIST_MH2NEURL','MEDHIST_MH3HEAD','MEDHIST_MH4CARD','MEDHIST_MH9ENDO','MEDHIST_MH14ALCH','MEDHIST_MH16SMOK','VITALS_VSWEIGHT','VITALS_VSHEIGHT','BMI_derived','VITALS_VSBPSYS','VITALS_VSBPDIA','VITALS_VSPULSE','PHYSICAL_PXABNORM','NEUROEXM_NXABNORM']
TYPE_IDS=[0,0,0,1,1,1,1,1,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4]

class TableDataset(Dataset):
    def __init__(self, labels, split_csv, table_csv, split, spatial=(160,196,160), stats=None, modality='mri', cache_dir=None):
        a=pd.read_csv(labels); s=pd.read_csv(split_csv); d=a.merge(s,on=['subject_id','sample_id'],validate='one_to_one')
        d=d[d.split==split].copy(); d=d[d.label.notna() & d.t1_path.notna()].sort_values(['subject_id','sample_id']).reset_index(drop=True)
        self.d=d; self.spatial=spatial; self.modality=str(modality).lower(); self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.modality=='pet': self.image_col='fdg_pet_path'
        elif self.modality=='mri_pet': self.image_col=None
        else: self.image_col='t1_path'
        if self.modality=='mri_pet': self.d=self.d[self.d['t1_path'].notna() & self.d['fdg_pet_path'].notna()].copy()
        else: self.d=self.d[self.d[self.image_col].notna()].copy()
        t=pd.read_csv(table_csv); t.columns=[str(c).lstrip('\ufeff') for c in t.columns]; t['DATE_KEY']=pd.to_datetime(t.VISDATE,errors='coerce').dt.strftime('%Y-%m-%d')
        t=t[['PTID','DATE_KEY']+TABLE_COLS].copy(); t[ TABLE_COLS]=t[TABLE_COLS].apply(pd.to_numeric,errors='coerce'); t=t.drop_duplicates(['PTID','DATE_KEY'])
        self.t={(str(r.PTID),str(r.DATE_KEY)):r for _,r in t.iterrows()}
        if stats is None:
            x=t[TABLE_COLS].to_numpy(np.float32); med=np.nanmedian(x,0); med=np.nan_to_num(med); imp=np.where(np.isnan(x),med,x); mean=imp.mean(0); std=imp.std(0); std[std<1e-6]=1
            self.stats={'med':med.astype('float32'),'mean':mean.astype('float32'),'std':std.astype('float32')}
        else:self.stats=stats
    def __len__(self): return len(self.d)
    def __getitem__(self,i):
        r=self.d.iloc[i]
        cp=self.cache_dir/f"{r.sample_id}.pt" if self.cache_dir else None
        if cp and cp.exists():
            z=torch.load(cp,map_location='cpu',weights_only=True)
            x=torch.cat([z['mri'].float(),z['pet'].float()],dim=0) if self.modality=='mri_pet' else z['pet' if self.modality=='pet' else 'mri'].float()
        elif self.modality=='mri_pet':
            x=torch.cat([_resize_volume(_load_volume(r.t1_path),self.spatial),_resize_volume(_load_volume(r.fdg_pet_path),self.spatial)],dim=0)
        else:
            path=str(r[self.image_col])
            if self.modality=='pet' and path.endswith('.pt'):
                obj=torch.load(path,map_location='cpu',weights_only=False); x=obj.get('prediction',obj.get('pet',obj))
                x=torch.as_tensor(x).float()
                while x.ndim > 4: x=x.squeeze(0)
                x=torch.nan_to_num(x,nan=0.0,posinf=1.0,neginf=-1.0)
                x=_resize_volume(x,self.spatial)
            else: x=_resize_volume(_load_volume(path),self.spatial)
        key=(str(r.subject_id),str(r.table_date_key))
        z=self.t.get(key); miss=np.ones(len(TABLE_COLS),np.float32) if z is None else z[TABLE_COLS].to_numpy(np.float32); miss=np.isnan(miss).astype(np.float32) if z is not None else miss
        vals=np.zeros(len(TABLE_COLS),np.float32) if z is None else z[TABLE_COLS].to_numpy(np.float32); vals=np.where(np.isnan(vals),self.stats['med'],vals); vals=(vals-self.stats['mean'])/self.stats['std']
        return {'input':x,'table':torch.tensor(vals),'table_mask':torch.tensor(miss),'label':torch.tensor(int(r.label),dtype=torch.long),'sample_id':str(r.sample_id)}

class BiCA(nn.Module):
    """One hierarchical bidirectional image-table interaction stage.

    The stage contains both directions:
      1) image queries table tokens and updates the image feature;
      2) table tokens query image tokens and update the table state.

    The returned table state is passed to the next BiCA stage, i.e.
    T0 -> BiCA3 -> T1 -> BiCA4 -> T2.
    """
    def __init__(self,ch,td=128,heads=4,image_residual_scale=0.10,table_update_scale=0.05,image_max_injection_ratio=0.02,table_max_update_ratio=0.10):
        super().__init__()
        if td % heads != 0: raise ValueError(f'token_dim ({td}) must be divisible by heads ({heads})')
        self.ch=int(ch); self.td=int(td); self.image_residual_scale=float(image_residual_scale); self.table_update_scale=float(table_update_scale); self.image_max_injection_ratio=float(image_max_injection_ratio); self.table_max_update_ratio=float(table_max_update_ratio)
        self.ip=nn.Conv3d(ch,td,1)
        self.image_query_norm=nn.LayerNorm(td); self.table_kv_norm=nn.LayerNorm(td); self.table_query_norm=nn.LayerNorm(td); self.image_kv_norm=nn.LayerNorm(td)
        self.image_reads_table=nn.MultiheadAttention(td,heads,batch_first=True)
        self.table_reads_image=nn.MultiheadAttention(td,heads,batch_first=True)
        self.image_delta_norm=nn.LayerNorm(td); self.table_delta_norm=nn.LayerNorm(td)
        self.out=nn.Conv3d(td,ch,1)
        nn.init.normal_(self.out.weight,mean=0.0,std=5e-3); nn.init.zeros_(self.out.bias)

    @staticmethod
    def _smooth_scale(ratio,max_ratio):
        x=ratio/float(max_ratio); eps=1e-6; safe_x=x.clamp_min(eps); candidate=torch.tanh(x)/safe_x
        return torch.where(x>eps,candidate,torch.ones_like(x))

    def _cap_image_delta(self,feature,delta):
        feature_norm=feature.float().flatten(1).norm(dim=1).clamp_min(1e-8); delta_norm=delta.float().flatten(1).norm(dim=1); raw=delta_norm/feature_norm
        scale=self._smooth_scale(raw,self.image_max_injection_ratio); delta=delta*scale[:,None,None,None,None].to(delta.dtype); final=delta.float().flatten(1).norm(dim=1)/feature_norm
        return delta,raw,final

    def _cap_table_delta(self,tokens,delta):
        token_norm=tokens.float().flatten(1).norm(dim=1).clamp_min(1e-8); delta_norm=delta.float().flatten(1).norm(dim=1); raw=delta_norm/token_norm
        scale=self._smooth_scale(raw,self.table_max_update_ratio); delta=delta*scale[:,None,None].to(delta.dtype); final=delta.float().flatten(1).norm(dim=1)/token_norm
        return delta,raw,final

    def forward(self,x,t,mask,observed,reliability):
        b,_,d,h,w=x.shape
        image_grid=self.ip(x); image_tokens=image_grid.flatten(2).transpose(1,2)

        # Table -> Image: image locations query clinical tokens.
        t_kv=self.table_kv_norm(t)
        image_from_table=self.image_reads_table(self.image_query_norm(image_tokens),t_kv,t_kv,key_padding_mask=mask.bool(),need_weights=False)[0]

        # Image -> Table: clinical tokens query the current image representation.
        image_kv=self.image_kv_norm(image_tokens)
        table_from_image=self.table_reads_image(self.table_query_norm(t),image_kv,image_kv,need_weights=False)[0]

        image_delta_tokens=self.image_delta_norm(image_from_table)
        delta_grid=image_delta_tokens.transpose(1,2).reshape(b,self.td,d,h,w)
        raw_image_delta=self.image_residual_scale*self.out(delta_grid)*reliability.to(x.dtype)[:, :, None, None, None]
        image_delta,_,_=self._cap_image_delta(x,raw_image_delta); x_next=x+image_delta

        observed_gate=observed.to(table_from_image.dtype).unsqueeze(-1)
        raw_table_delta=self.table_update_scale*self.table_delta_norm(table_from_image)*observed_gate
        table_delta,_,_=self._cap_table_delta(t,raw_table_delta); t_next=t+table_delta
        return x_next,t_next

class Exp08(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__(); self.back=resnet50(shortcut_type='B',in_channels=in_channels); self.table=ClinicalTableEncoder(25,TYPE_IDS,token_dim=128,depth=2,heads=4,ff_mult=4); self.bica3=BiCA(1024); self.bica4=BiCA(2048); self.gvit=GlobalViT3D(2048,token_dim=256,depth=3,heads=8,bottleneck_shape=(5,6,5)); self.gproj=nn.Conv3d(2048,2048,1); self.gate=nn.Sequential(nn.Conv3d(2048,2048,1),nn.Sigmoid()); self.head=nn.Sequential(nn.Linear(1024+2048+2048+128,512),nn.GELU(),nn.Dropout(.3),nn.Linear(512,3))

    def _subject_from_tokens(self,tokens,observed,reliability):
        weights=observed.to(tokens.dtype).unsqueeze(-1); pooled=(tokens*weights).sum(dim=1)/weights.sum(dim=1).clamp_min(1.0)
        return self.table.subject_mlp(torch.cat([pooled,reliability.to(pooled.dtype)],dim=1))

    def forward(self,x,table,mask):
        b=self.back; x=b.relu(b.bn1(b.conv1(x))); x=b.maxpool(x); f1=b.layer1(x); f2=b.layer2(f1); f3=b.layer3(f2); f4=b.layer4(f3); ts=self.table(table,mask); table_tokens=ts['tokens']; f3,table_tokens=self.bica3(f3,table_tokens,ts['key_padding_mask'],ts['effective_observed'],ts['reliability']); f4,table_tokens=self.bica4(f4,table_tokens,ts['key_padding_mask'],ts['effective_observed'],ts['reliability']); g=self.gvit(F.adaptive_avg_pool3d(f4,(5,6,5))); g=F.interpolate(g,size=f4.shape[-3:],mode='trilinear',align_corners=False); f4=f4+self.gate(f4)*self.gproj(g); z3=F.adaptive_avg_pool3d(f3,1).flatten(1); z4=F.adaptive_avg_pool3d(f4,1).flatten(1); zg=F.adaptive_avg_pool3d(g,1).flatten(1); zt=self._subject_from_tokens(table_tokens,ts['effective_observed'],ts['reliability']); return self.head(torch.cat([z3,z4,zg,zt],1))

def metrics(y,p,prob):
 cm=np.zeros((3,3),int)
 for a,b in zip(y,p):cm[a,b]+=1
 rec=[];pre=[];f1=[]
 for i in range(3):
  tp=cm[i,i]; fn=cm[i].sum()-tp; fp=cm[:,i].sum()-tp; r=tp/(tp+fn) if tp+fn else 0; q=tp/(tp+fp) if tp+fp else 0; rec.append(r);pre.append(q);f1.append(2*r*q/(r+q) if r+q else 0)
 try: auc=float(roc_auc_score(y,np.asarray(prob),multi_class='ovr',average='macro'))
 except: auc=None
 return {'accuracy':float(np.trace(cm)/cm.sum()),'balanced_accuracy':float(np.mean(rec)),'macro_precision':float(np.mean(pre)),'macro_recall':float(np.mean(rec)),'macro_f1':float(np.mean(f1)),'macro_auc':auc,'confusion_matrix':cm.tolist(),'per_class_recall':rec}

def run_epoch(m,dl,opt,dev,criterion):
 train=opt is not None; m.train(train); ys=[];ps=[];probs=[];losses=[]
 for q in dl:
  x=q['input'].to(dev); t=q['table'].to(dev); mm=q['table_mask'].to(dev); y=q['label'].to(dev); logits=m(x,t,mm).float(); loss=criterion(logits,y)
  if train: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
  losses.append(loss.item()); ys+=y.cpu().tolist(); ps+=logits.argmax(1).cpu().tolist(); probs+=torch.softmax(logits,1).detach().cpu().tolist()
 return float(np.mean(losses)),metrics(ys,ps,probs)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); c=Path(a.config); cfg=json.loads(c.read_text()) if c.suffix=='.json' else __import__('yaml').safe_load(c); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed']); torch.set_num_threads(int(cfg.get('cpu_threads',8))); torch.set_num_interop_threads(2); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True); mod=cfg.get('modality','mri'); cache=cfg.get('cache_dir'); train=TableDataset(cfg['labels_csv'],cfg['split_csv'],cfg['table_csv'],'train',modality=mod,cache_dir=cache); val=TableDataset(cfg['labels_csv'],cfg['split_csv'],cfg['table_csv'],'val',stats=train.stats,modality=mod,cache_dir=cache); final=TableDataset(cfg['labels_csv'],cfg['split_csv'],cfg['table_csv'],'final_val',stats=train.stats,modality=mod,cache_dir=cache); dl=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers']); vl=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers']); fl=DataLoader(final,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers']); m=Exp08(in_channels=2 if mod=='mri_pet' else 1).to(dev); w=torch.tensor(cfg['class_weight'],dtype=torch.float32,device=dev); opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); crit=nn.CrossEntropyLoss(weight=w); best=-1; hist=[]
 print('Exp08',dev,'train/val/final',len(train),len(val),len(final),'table',len(TABLE_COLS),flush=True)
 start_ep=0
 resume=cfg.get('resume_from')
 init_from=cfg.get('init_from')
 if init_from and Path(init_from).exists():
  state=torch.load(init_from,map_location=dev,weights_only=False); m.load_state_dict(state['model_state_dict']); print('INITIALIZED_FROM',init_from,'epoch',state.get('epoch'),flush=True)
 if resume and Path(resume).exists():
  state=torch.load(resume,map_location=dev,weights_only=False); m.load_state_dict(state['model_state_dict']); start_ep=int(state.get('epoch',0)); print('RESUMED',resume,'epoch',start_ep,flush=True)
  if 'optimizer_state_dict' in state: opt.load_state_dict(state['optimizer_state_dict'])
  previous_metrics=out/'best_metrics.json'
  if previous_metrics.exists():
   best=float(json.loads(previous_metrics.read_text())['val']['macro_f1'])
 for ep in range(start_ep+1,cfg['epochs']+1):
  tl,tm=run_epoch(m,dl,opt,dev,crit); vlss,vm=run_epoch(m,vl,None,dev,crit); hist.append({'epoch':ep,'train_loss':tl,'val_loss':vlss,'train':tm,'val':vm}); print(ep,tl,vm['macro_f1'],vm['accuracy'],flush=True)
  state={'model_state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':ep,'config':cfg}
  torch.save(state,out/'last.pt')
  if vm['macro_f1']>best: best=vm['macro_f1']; torch.save(state,out/'best.pt'); (out/'best_metrics.json').write_text(json.dumps({'epoch':ep,'val':vm},indent=2))
 ck=torch.load(out/'best.pt',map_location=dev); m.load_state_dict(ck['model_state_dict']); _,fm=run_epoch(m,fl,None,dev,crit); (out/'metrics.json').write_text(json.dumps({'best_epoch':ck['epoch'],'best_val':json.loads((out/'best_metrics.json').read_text())['val'],'final_val':fm,'history':hist},indent=2)); print('FINAL',fm,flush=True)
if __name__=='__main__':main()
