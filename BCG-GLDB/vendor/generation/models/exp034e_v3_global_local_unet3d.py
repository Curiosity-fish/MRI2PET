"""Exp034E_v3: bottleneck global Mamba with gated multi-resolution local fusion."""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn.functional as F
from torch import nn

from models.exp034a_basic_unet3d import BasicUNet3D
from models.global_local_mamba_patch_denoiser import AxialBiMambaBlock, DualGateFusion3D


class GlobalMambaContext3D(nn.Module):
    def __init__(self, channels, time_dim, token_dim=128, d_state=8, d_conv=3, expand=1, line_chunk_size=256, depth=1):
        super().__init__()
        self.to_tokens = nn.Conv3d(channels, token_dim, 1)
        self.blocks = nn.ModuleList([AxialBiMambaBlock(dim=token_dim, condition_dim=time_dim, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=True, active_axes=('x', 'y', 'z'), line_chunk_size=line_chunk_size, gradient_checkpointing=True) for _ in range(depth)])
        self.to_context = nn.Conv3d(token_dim, channels, 1)
        nn.init.zeros_(self.to_context.weight); nn.init.zeros_(self.to_context.bias)

    def forward(self, x, time_emb):
        token = self.to_tokens(x); zeros = torch.zeros_like(time_emb); reliability = token.new_zeros((token.shape[0], 1))
        for block in self.blocks:
            token = block(token, base_condition=time_emb, table_condition=zeros, table_reliability=reliability)
        return self.to_context(token)


class GlobalLocalMambaUNet3D(BasicUNet3D):
    def __init__(self, *args, mamba_token_dim=128, mamba_d_state=8, mamba_d_conv=3, mamba_expand=1, mamba_line_chunk_size=256, mamba_depth=1, **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = (int(v) for v in kwargs.get('channels', (96, 192, 384, 768)))
        common = dict(time_dim=int(kwargs.get('time_dim', 384)), token_dim=int(mamba_token_dim), d_state=int(mamba_d_state), d_conv=int(mamba_d_conv), expand=int(mamba_expand), line_chunk_size=int(mamba_line_chunk_size), depth=int(mamba_depth))
        self.global_context = GlobalMambaContext3D(c3, **common)
        self.fuse_mid = DualGateFusion3D(c3, c3, int(kwargs.get('norm_groups', 8)))
        self.context_to_dec2 = nn.Conv3d(c3, c2, 1)
        self.context_to_dec1 = nn.Conv3d(c3, c1, 1)
        self.fuse_dec2 = DualGateFusion3D(c2, c2, int(kwargs.get('norm_groups', 8)))
        self.fuse_dec1 = DualGateFusion3D(c1, c1, int(kwargs.get('norm_groups', 8)))
        for layer in (self.context_to_dec2, self.context_to_dec1):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, x1, t):
        te = self.time_embedding(t)
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te)
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te)
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te)
        h = self.mid2(self.mid1(self.down3(s2), te), te)
        context = self.global_context(h, te)
        h = self.fuse_mid(h, context)
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], 1), te), te)
        guide2 = F.interpolate(self.context_to_dec2(context), size=h.shape[-3:], mode='trilinear', align_corners=False)
        h = self.fuse_dec2(h, guide2)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], 1), te), te)
        guide1 = F.interpolate(self.context_to_dec1(context), size=h.shape[-3:], mode='trilinear', align_corners=False)
        h = self.fuse_dec1(h, guide1)
        h = self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], 1), te), te)
        return self.out(h)
