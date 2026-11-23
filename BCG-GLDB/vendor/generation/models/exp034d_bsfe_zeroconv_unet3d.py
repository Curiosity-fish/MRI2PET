from __future__ import annotations

from typing import Sequence
import torch
from torch import nn
from models.exp034a_basic_unet3d import TimeEmbedding, TimeResBlock3D


class BSFEZeroConvUNet3D(nn.Module):
    """Exp034A U-Net with frozen/trainable BSFE pyramid zero-conv adapters."""
    def __init__(self, in_channels=4, out_channels=4, channels=(96,192,384,768),
                 bsfe_channels=(48,96,192,384,192,96,48), time_dim=384,
                 norm_groups=8, dropout=0.0):
        super().__init__(); c0,c1,c2,c3=map(int,channels)
        self.time_embedding=TimeEmbedding(time_dim)
        self.input=nn.Conv3d(in_channels,c0,3,padding=1)
        self.enc0a=TimeResBlock3D(c0,c0,time_dim,norm_groups,dropout); self.enc0b=TimeResBlock3D(c0,c0,time_dim,norm_groups,dropout)
        self.down1=nn.Conv3d(c0,c1,3,stride=2,padding=1)
        self.enc1a=TimeResBlock3D(c1,c1,time_dim,norm_groups,dropout); self.enc1b=TimeResBlock3D(c1,c1,time_dim,norm_groups,dropout)
        self.down2=nn.Conv3d(c1,c2,3,stride=2,padding=1)
        self.enc2a=TimeResBlock3D(c2,c2,time_dim,norm_groups,dropout); self.enc2b=TimeResBlock3D(c2,c2,time_dim,norm_groups,dropout)
        self.down3=nn.Conv3d(c2,c3,3,stride=2,padding=1)
        self.mid1=TimeResBlock3D(c3,c3,time_dim,norm_groups,dropout); self.mid2=TimeResBlock3D(c3,c3,time_dim,norm_groups,dropout)
        self.up2=nn.ConvTranspose3d(c3,c2,4,stride=2,padding=1); self.dec2a=TimeResBlock3D(c2*2,c2,time_dim,norm_groups,dropout); self.dec2b=TimeResBlock3D(c2,c2,time_dim,norm_groups,dropout)
        self.up1=nn.ConvTranspose3d(c2,c1,4,stride=2,padding=1); self.dec1a=TimeResBlock3D(c1*2,c1,time_dim,norm_groups,dropout); self.dec1b=TimeResBlock3D(c1,c1,time_dim,norm_groups,dropout)
        self.up0=nn.ConvTranspose3d(c1,c0,4,stride=2,padding=1); self.dec0a=TimeResBlock3D(c0*2,c0,time_dim,norm_groups,dropout); self.dec0b=TimeResBlock3D(c0,c0,time_dim,norm_groups,dropout)
        stage_channels=(c0,c1,c2,c3,c2,c1,c0)
        self.bsfe_zero=nn.ModuleList([nn.Conv3d(int(i),int(o),1) for i,o in zip(bsfe_channels,stage_channels)])
        for layer in self.bsfe_zero: nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.out=nn.Sequential(nn.GroupNorm(min(norm_groups,c0),c0),nn.SiLU(),nn.Conv3d(c0,out_channels,3,padding=1))

    def _inject(self,x,pyramid,idx):
        return x + self.bsfe_zero[idx](pyramid[idx])

    def forward(self,x1,t,pyramid):
        te=self.time_embedding(t)
        if len(pyramid) != 4:
            raise ValueError(f'Expected four BSFE pyramid levels, got {len(pyramid)}')
        # The encoder has four levels; decoder stages reuse the matching
        # encoder features in reverse order.
        stage_pyramid = [pyramid[0], pyramid[1], pyramid[2], pyramid[3], pyramid[2], pyramid[1], pyramid[0]]
        s0=self.enc0b(self.enc0a(self.input(x1),te),te); s0=self._inject(s0,stage_pyramid,0)
        s1=self.enc1b(self.enc1a(self.down1(s0),te),te); s1=self._inject(s1,stage_pyramid,1)
        s2=self.enc2b(self.enc2a(self.down2(s1),te),te); s2=self._inject(s2,stage_pyramid,2)
        h=self.mid2(self.mid1(self.down3(s2),te),te); h=self._inject(h,stage_pyramid,3)
        h=self.dec2b(self.dec2a(torch.cat([self.up2(h),s2],1),te),te); h=self._inject(h,stage_pyramid,4)
        h=self.dec1b(self.dec1a(torch.cat([self.up1(h),s1],1),te),te); h=self._inject(h,stage_pyramid,5)
        h=self.dec0b(self.dec0a(torch.cat([self.up0(h),s0],1),te),te); h=self._inject(h,stage_pyramid,6)
        return self.out(h)
