"""Exp034E: direct x1-to-Mamba conditioning at three low-resolution U-Net stages."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.exp034a_basic_unet3d import BasicUNet3D
from models.global_local_mamba_patch_denoiser import AxialBiMambaBlock


class X1MambaFusion(nn.Module):
    """Encode the same x1 state at one U-Net resolution and add it residually."""

    def __init__(self, in_channels: int, out_channels: int, target_shape: Sequence[int], time_dim: int,
                 token_dim: int = 128, d_state: int = 8, d_conv: int = 3, expand: int = 1,
                 line_chunk_size: int = 512, depth: int = 1):
        super().__init__()
        self.target_shape = tuple(int(v) for v in target_shape)
        self.project = nn.Conv3d(in_channels, out_channels, 1)
        self.to_mamba = nn.Conv3d(out_channels, token_dim, 1)
        self.blocks = nn.ModuleList([
            AxialBiMambaBlock(
                dim=token_dim, condition_dim=time_dim, d_state=d_state, d_conv=d_conv,
                expand=expand, bidirectional=True, active_axes=("x", "y", "z"),
                line_chunk_size=line_chunk_size, gradient_checkpointing=True,
            ) for _ in range(int(depth))
        ])
        self.from_mamba = nn.Conv3d(token_dim, out_channels, 1)
        self.fusion = nn.Conv3d(out_channels, out_channels, 1)
        nn.init.normal_(self.fusion.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x1: torch.Tensor, time_emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x1, size=self.target_shape, mode="trilinear", align_corners=False)
        x = self.project(x)
        x = self.to_mamba(x)
        table_condition = torch.zeros_like(time_emb)
        reliability = x.new_zeros((x.shape[0], 1))
        for block in self.blocks:
            x = block(x, base_condition=time_emb, table_condition=table_condition, table_reliability=reliability)
        update = self.fusion(self.from_mamba(x))
        return target + update


class MambaLowResUNet3D(BasicUNet3D):
    """Exp034A backbone with joint x1-Mamba fusion at s2, h, and decoder dec2."""

    def __init__(self, *args, latent_shape: Sequence[int] = (40, 48, 40), mamba_token_dim: int = 128,
                 mamba_d_state: int = 8, mamba_d_conv: int = 3, mamba_expand: int = 1,
                 mamba_line_chunk_size: int = 512, mamba_depth: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = (int(v) for v in kwargs.get("channels", (96, 192, 384, 768)))
        quarter = tuple(int(v) // 4 for v in latent_shape)
        eighth = tuple(int(v) // 8 for v in latent_shape)
        common = dict(time_dim=int(kwargs.get("time_dim", 384)), token_dim=mamba_token_dim,
                      d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
                      line_chunk_size=mamba_line_chunk_size, depth=mamba_depth)
        self.mamba_enc_s2 = X1MambaFusion(4, c2, quarter, **common)
        self.mamba_mid = X1MambaFusion(4, c3, eighth, **common)
        self.mamba_dec_s2 = X1MambaFusion(4, c2, quarter, **common)

    def forward(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embedding(t)
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te)
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te)
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te)
        s2 = self.mamba_enc_s2(x1, te, s2)
        h = self.mid2(self.mid1(self.down3(s2), te), te)
        h = self.mamba_mid(x1, te, h)
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], dim=1), te), te)
        h = self.mamba_dec_s2(x1, te, h)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], dim=1), te), te)
        h = self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], dim=1), te), te)
        return self.out(h)
