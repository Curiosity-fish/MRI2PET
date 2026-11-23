"""Exp034M curriculum U-Net: A_v3 backbone plus staged table/BSFE/Mamba guidance."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from models.exp034a_v2_global_attention_unet3d import ResBlock3D, GlobalSelfAttention3D, SafeGroupNorm, TimeToInput
from models.exp034a_basic_unet3d import TimeEmbedding
from models.exp034c_table_bica_unet3d import BidirectionalTableStage
from models.global_local_mamba_patch_denoiser import AxialBiMambaBlock, DualGateFusion3D

class MambaContext(nn.Module):
    def __init__(self, channels, time_dim, token_dim=128):
        super().__init__(); self.to_tokens=nn.Conv3d(channels,token_dim,1)
        self.block=AxialBiMambaBlock(dim=token_dim,condition_dim=time_dim,d_state=8,d_conv=3,expand=1,bidirectional=True,active_axes=('x','y','z'),line_chunk_size=256,gradient_checkpointing=True)
        self.to_context=nn.Conv3d(token_dim,channels,1); nn.init.zeros_(self.to_context.weight); nn.init.zeros_(self.to_context.bias)
    def forward(self,x,te):
        tok=self.to_tokens(x); rel=tok.new_zeros((tok.shape[0],1)); z=torch.zeros_like(te)
        return self.to_context(self.block(tok,base_condition=te,table_condition=z,table_reliability=rel))

class Exp034MCurriculumUNet3D(nn.Module):
    def __init__(self, table_features=45, feature_type_ids=None, channels=(64,128,256,512), time_dim=384, groups=8, heads=4, table_dim=128, deepest_channels=None):
        super().__init__(); c0,c1,c2,c3=map(int,channels); cd=int(deepest_channels or c3); self.channels=(c0,c1,c2,c3,cd); self.time=TimeToInput(time_dim,4); self.condition_time=TimeEmbedding(time_dim)
        self.stem=ResBlock3D(4,c0,groups); self.downs=nn.ModuleList(); self.downconvs=nn.ModuleList()
        for a,b in ((c0,c1),(c1,c2),(c2,c3)):
            self.downs.append(nn.ModuleList([ResBlock3D(a,a,groups),ResBlock3D(a,a,groups),GlobalSelfAttention3D(a,heads,groups)])); self.downconvs.append(nn.Conv3d(a,b,4,2,1))
        self.mid=nn.ModuleList([ResBlock3D(c3,cd,groups),ResBlock3D(cd,c3,groups)])
        self.ups=nn.ModuleList(); self.upconvs=nn.ModuleList()
        for a,b in ((c3,c2),(c2,c1),(c1,c0)):
            self.upconvs.append(nn.ConvTranspose3d(a,b,4,2,1)); self.ups.append(nn.ModuleList([ResBlock3D(b*2,b,groups),ResBlock3D(b,b,groups),GlobalSelfAttention3D(b,heads,groups)]))
        self.out=nn.Sequential(SafeGroupNorm(c0,groups),nn.SiLU(),nn.Conv3d(c0,4,3,padding=1))
        self.table_stages=nn.ModuleList([BidirectionalTableStage(c,table_dim,time_dim,g,heads,0.05) for c,g in ((c0,(10,12,10)),(c1,(10,12,10)),(c2,(10,12,10)),(c1,(10,12,10)))])
        self.table_proj=nn.ModuleList([nn.Conv3d(c,c,1) for c in (c0,c1,c1,c0)])
        self.bsfe_proj=nn.ModuleList([nn.Conv3d(x,y,1) for x,y in zip((48,96,192,384),(c0,c1,c2,c3))])
        self.bsfe_proj_up=nn.ModuleList([nn.Conv3d(x,y,1) for x,y in zip((192,96,48),(c2,c1,c0))])
        self.mamba=MambaContext(c3,time_dim,128); self.fuse_mid=DualGateFusion3D(c3,c3,groups); self.ctx2=nn.Conv3d(c3,c2,1); self.ctx1=nn.Conv3d(c3,c1,1); self.fuse2=DualGateFusion3D(c2,c2,groups); self.fuse1=DualGateFusion3D(c1,c1,groups)
        for x in (self.ctx2,self.ctx1): nn.init.zeros_(x.weight); nn.init.zeros_(x.bias)
        for x in list(self.bsfe_proj) + list(self.bsfe_proj_up): nn.init.zeros_(x.weight); nn.init.zeros_(x.bias)
        self.table_dim=table_dim; self.table_features=table_features
    def _table(self,x,idx,tok,mask,te): return self.table_stages[idx](x,tok,mask,te)
    def forward(self,x1,t,table=None,table_mask=None,pyramid=None,stage=1):
        te_input=self.time(t).to(x1.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1); te=self.condition_time(t); x=self.stem(x1+te_input); tok=None
        if stage>=2:
            # Lazy tokenizer keeps the module compatible with variable table width.
            if not hasattr(self,'tokenizer'): raise RuntimeError('table tokenizer was not initialized')
            tok,kpm,_=self.tokenizer(table,table_mask)
        skips=[]
        for i,(blocks,down) in enumerate(zip(self.downs,self.downconvs)):
            x=blocks[0](x)
            if stage>=2 and i<2: x,tok=self._table(x,i,tok,kpm,te)
            x=blocks[2](blocks[1](x));
            if stage>=3 and pyramid is not None: x=x+self.bsfe_proj[i](pyramid[i])
            skips.append(x); x=down(x)
        x=self.mid[1](self.mid[0](x))
        if stage>=3 and pyramid is not None: x=x+self.bsfe_proj[3](pyramid[3])
        if stage>=4: ctx=self.mamba(x,te.squeeze(-1).squeeze(-1).squeeze(-1)); x=self.fuse_mid(x,ctx)
        for j,(up,blocks,skip) in enumerate(zip(self.upconvs,self.ups,reversed(skips))):
            x=up(x); x=F.interpolate(x,size=skip.shape[-3:],mode='trilinear',align_corners=False) if x.shape[-3:]!=skip.shape[-3:] else x; x=blocks[0](torch.cat((x,skip),1))
            if stage>=2 and j<2: x,tok=self._table(x,2+j,tok,kpm,te)
            x=blocks[2](blocks[1](x));
            if stage>=3 and pyramid is not None: x=x+self.bsfe_proj_up[j](pyramid[2-j])
            if stage>=4 and j<2:
                guide=F.interpolate((self.ctx2 if j==0 else self.ctx1)(ctx),size=x.shape[-3:],mode='trilinear',align_corners=False); x=(self.fuse2 if j==0 else self.fuse1)(x,guide)
        return self.out(x)
