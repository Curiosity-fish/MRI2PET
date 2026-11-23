"""Basic time-conditioned 3D U-Net used by Exp034A.

This module deliberately has no MRI encoder, table path, attention, or bridge
residual head.  It maps the MRI-shaped latent endpoint x1 directly to PET
latent z0 and only receives a scalar diffusion-time embedding in addition.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.net(emb)


class TimeResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int, dropout: float) -> None:
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class BasicUNet3D(nn.Module):
    """Four-resolution residual U-Net for a [B, 4, 40, 48, 40] latent volume."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 channels: tuple[int, int, int, int] = (96, 192, 384, 768),
                 time_dim: int = 384, norm_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.time_embedding = TimeEmbedding(time_dim)
        self.input = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.enc0a = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.enc0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1a = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.enc1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2a = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.enc2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid1 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.mid2 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.dec2a = TimeResBlock3D(c2 * 2, c2, time_dim, norm_groups, dropout)
        self.dec2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)
        self.dec1a = TimeResBlock3D(c1 * 2, c1, time_dim, norm_groups, dropout)
        self.dec1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.up0 = nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1)
        self.dec0a = TimeResBlock3D(c0 * 2, c0, time_dim, norm_groups, dropout)
        self.dec0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.out = nn.Sequential(nn.GroupNorm(min(norm_groups, c0), c0), nn.SiLU(), nn.Conv3d(c0, out_channels, 3, padding=1))

    def forward(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_embedding(t)
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te)
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te)
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te)
        h = self.mid2(self.mid1(self.down3(s2), te), te)
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h), s2], dim=1), te), te)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h), s1], dim=1), te), te)
        h = self.dec0b(self.dec0a(torch.cat([self.up0(h), s0], dim=1), te), te)
        return self.out(h)
