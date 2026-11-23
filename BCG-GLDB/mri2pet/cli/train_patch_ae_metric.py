"""Train Patch-AE with a volume L1 objective and select by validation mean SSIM."""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from mri2pet.data.paired_pet import PairedFDGPETDataset
from mri2pet.data.patches import extract_patches
from mri2pet.models.generation import SharedPatchAE

def image_metrics(pred, target):
    """Per-volume full-grid metrics; separable Gaussian equals the 3-D SSIM kernel."""
    pred, target = pred.float(), target.float()
    difference = pred-target
    mae = difference.abs().mean()
    mse = difference.square().mean()
    rmse = mse.sqrt()
    psnr = 10*torch.log10(torch.tensor(4.0,device=pred.device)/mse.clamp_min(1e-12))
    x = torch.arange(11,device=pred.device,dtype=torch.float32)-5
    kernel = torch.exp(-(x*x)/(2*1.5**2)); kernel = kernel/kernel.sum()
    def blur(t):
        t=F.conv3d(t,kernel.view(1,1,11,1,1),padding=(5,0,0))
        t=F.conv3d(t,kernel.view(1,1,1,11,1),padding=(0,5,0))
        return F.conv3d(t,kernel.view(1,1,1,1,11),padding=(0,0,5))
    mu_p,mu_t=blur(pred),blur(target)
    var_p=blur(pred*pred)-mu_p*mu_p
    var_t=blur(target*target)-mu_t*mu_t
    cov=blur(pred*target)-mu_p*mu_t
    ssim=(((2*mu_p*mu_t+0.0004)*(2*cov+0.0036))/((mu_p.square()+mu_t.square()+0.0004)*(var_p+var_t+0.0036))).mean()
    return dict(psnr=float(psnr),ssim=float(ssim),mae=float(mae),rmse=float(rmse))

def train_batch(ae, pet, optimizer, chunk_size):
    """Accumulate exact volume-mean L1 gradients over equally sized patch chunks."""
    patches=extract_patches(pet,ae.patch_coords,ae.patch_shape)
    optimizer.zero_grad(set_to_none=True)
    total=0.0
    for start in range(0,64,chunk_size):
        part=patches[:,start:start+chunk_size]
        x=part.reshape(-1,1,*ae.patch_shape)
        reconstructed=ae.net.decoder(ae.net.encode(x))
        loss=F.l1_loss(reconstructed,x)*(part.shape[1]/64.0)
        loss.backward()
        total += float(loss.detach())
    optimizer.step()
    return total

def run(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    root=Path(cfg['data']['root'])
    output=Path(cfg['output_dir']); output.mkdir(parents=True,exist_ok=True)
    if (output/'history.jsonl').exists() or (output/'best_patch_ae.pt').exists():
        raise FileExistsError('Output already contains a run; choose a new output directory.')
    random.seed(cfg.get('seed',42)); np.random.seed(cfg.get('seed',42)); torch.manual_seed(cfg.get('seed',42))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this training run')
    torch.cuda.manual_seed_all(cfg.get('seed',42))
    device=torch.device('cuda')
    image_shape=tuple(cfg['data']['image_shape']); patch_shape=tuple(cfg['data']['patch_shape'])
    ae=SharedPatchAE(image_shape=image_shape,patch_shape=patch_shape,latent_channels=4,
                     base_channels=int(cfg['model']['base_channels']),
                     chunk_size=int(cfg['training']['patch_chunk_size'])).to(device)
    datasets={name:PairedFDGPETDataset(pairs_csv=root/cfg['data']['pairs_csv'],
             split_csv=root/cfg['data']['split_csv'],split=name,fold=None,spatial_size=image_shape,
             normalize_targets=False) for name in ('train','val')}
    assert not(set(datasets['train'].df.subject_id.astype(str)) & set(datasets['val'].df.subject_id.astype(str)))
    loaders={name:DataLoader(ds,batch_size=int(cfg['training']['batch_size']),shuffle=name=='train',
             num_workers=int(cfg['training']['num_workers']),pin_memory=True) for name,ds in datasets.items()}
    optimizer=torch.optim.AdamW(ae.parameters(),lr=float(cfg['training']['lr']),
                                weight_decay=float(cfg['training'].get('weight_decay',0.0)))
    best=-float('inf'); best_epoch=0
    print(json.dumps({'event':'started','device':torch.cuda.get_device_name(0),'training_samples':len(datasets['train']),
                      'validation_samples':len(datasets['val']),'epochs':cfg['training']['epochs'],
                      'selection':'validation_mean_ssim','patch_chunk_size':cfg['training']['patch_chunk_size']}),flush=True)
    for epoch in range(1,int(cfg['training']['epochs'])+1):
        started=time.time(); ae.train(); total=0.0; seen=0
        for batch in loaders['train']:
            pet=batch['target'].to(device,non_blocking=True)
            loss=train_batch(ae,pet,optimizer,int(cfg['training']['patch_chunk_size']))
            if not math.isfinite(loss): raise RuntimeError('Non-finite training loss')
            total+=loss*len(pet); seen+=len(pet)
            if seen%100==0 or seen==len(datasets['train']):
                print(json.dumps({'event':'train_progress','epoch':epoch,'samples':seen,'loss':total/seen}),flush=True)
        ae.eval(); results=[]
        with torch.no_grad():
            for batch in loaders['val']:
                pet=batch['target'].to(device,non_blocking=True)
                pred=ae(pet)
                for i in range(len(pet)): results.append(image_metrics(pred[i:i+1],pet[i:i+1]))
        metrics={k:float(np.mean([r[k] for r in results])) for k in ('psnr','ssim','mae','rmse')}
        if not all(math.isfinite(v) for v in metrics.values()): raise RuntimeError('Non-finite validation metric')
        rec={'event':'epoch_end','epoch':epoch,'train_l1':total/seen,'val':metrics,'elapsed_seconds':round(time.time()-started,2)}
        payload={'model':ae.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'epoch':epoch,'validation':metrics}
        torch.save(payload,output/'last_patch_ae.pt')
        if metrics['ssim']>best:
            best=metrics['ssim']; best_epoch=epoch
            torch.save(payload,output/'best_patch_ae.pt')
            (output/'best_validation.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
        rec['best_epoch']=best_epoch
        with (output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(parser.parse_args().config)
