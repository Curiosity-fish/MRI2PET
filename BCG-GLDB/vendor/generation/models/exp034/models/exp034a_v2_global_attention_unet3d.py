"""Exp034A_v2: MRI-AE latent U-Net with full 3D self-attention at each stage."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    import xformers.ops as xops
except ImportError as exc:  # Full-resolution global attention is not viable with the math fallback.
    raise RuntimeError('Exp034A_v2 requires xformers memory_efficient_attention.') from exc


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        groups = min(int(groups), int(channels))
        while groups > 1 and channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block1 = nn.Sequential(SafeGroupNorm(in_channels, groups), nn.SiLU(), nn.Conv3d(in_channels, out_channels, 3, padding=1))
        self.block2 = nn.Sequential(SafeGroupNorm(out_channels, groups), nn.SiLU(), nn.Dropout3d(dropout) if dropout else nn.Identity(), nn.Conv3d(out_channels, out_channels, 3, padding=1))
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.skip(x)


class GlobalSelfAttention3D(nn.Module):
    """Exact voxel-to-voxel multi-head attention, using memory-efficient SDPA kernels."""
    def __init__(self, channels: int, heads: int, groups: int) -> None:
        super().__init__()
        heads = min(int(heads), int(channels))
        while heads > 1 and channels % heads:
            heads -= 1
        self.heads = heads
        self.norm = SafeGroupNorm(channels, groups)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        tokens = depth * height * width
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        # xFormers computes exact global attention without materializing N x N maps.
        q = q.reshape(batch, self.heads, channels // self.heads, tokens).permute(0, 3, 1, 2).contiguous()
        k = k.reshape(batch, self.heads, channels // self.heads, tokens).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(batch, self.heads, channels // self.heads, tokens).permute(0, 3, 1, 2).contiguous()
        out = xops.memory_efficient_attention(q, k, v)
        out = out.permute(0, 2, 3, 1).reshape(batch, channels, depth, height, width)
        return x + self.proj(out)


class TimeToInput(nn.Module):
    def __init__(self, time_dim: int, channels: int) -> None:
        super().__init__()
        self.time_dim = int(time_dim)
        self.net = nn.Sequential(nn.Linear(time_dim, time_dim * 2), nn.SiLU(), nn.Linear(time_dim * 2, channels))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        phase = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat((phase.sin(), phase.cos()), dim=1)
        if emb.shape[1] < self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.shape[1]))
        return self.net(emb)


class DownStage(nn.Module):
    def __init__(self, channels: int, next_channels: int, groups: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.res1 = ResBlock3D(channels, channels, groups, dropout)
        self.res2 = ResBlock3D(channels, channels, groups, dropout)
        self.attn = GlobalSelfAttention3D(channels, heads, groups)
        self.down = nn.Conv3d(channels, next_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.attn(self.res2(self.res1(x)))
        return skip, self.down(skip)


class UpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, groups: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, skip_channels, 4, stride=2, padding=1)
        self.res1 = ResBlock3D(skip_channels * 2, skip_channels, groups, dropout)
        self.res2 = ResBlock3D(skip_channels, skip_channels, groups, dropout)
        self.attn = GlobalSelfAttention3D(skip_channels, heads, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
        return self.attn(self.res2(self.res1(torch.cat((x, skip), dim=1))))


class Exp034AV2GlobalAttentionUNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 4, channels: tuple[int, int, int, int] = (32, 64, 128, 256), time_dim: int = 128, norm_groups: int = 8, attention_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        c0, c1, c2, c3 = (int(value) for value in channels)
        self.time_to_input = TimeToInput(time_dim, in_channels)
        self.stem = ResBlock3D(in_channels, c0, norm_groups, dropout)
        self.down0 = DownStage(c0, c1, norm_groups, attention_heads, dropout)
        self.down1 = DownStage(c1, c2, norm_groups, attention_heads, dropout)
        self.down2 = DownStage(c2, c3, norm_groups, attention_heads, dropout)
        self.mid1 = ResBlock3D(c3, c3, norm_groups, dropout)
        self.mid2 = ResBlock3D(c3, c3, norm_groups, dropout)
        self.up2 = UpStage(c3, c2, norm_groups, attention_heads, dropout)
        self.up1 = UpStage(c2, c1, norm_groups, attention_heads, dropout)
        self.up0 = UpStage(c1, c0, norm_groups, attention_heads, dropout)
        self.out = nn.Sequential(SafeGroupNorm(c0, norm_groups), nn.SiLU(), nn.Conv3d(c0, out_channels, 3, padding=1))

    def forward(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x1 + self.time_to_input(t).to(dtype=x1.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.stem(x)
        s0, x = self.down0(x)
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        x = self.mid2(self.mid1(x))
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.out(x)
