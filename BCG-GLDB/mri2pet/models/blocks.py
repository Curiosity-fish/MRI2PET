from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def safe_num_groups(channels: int, requested: int) -> int:
    groups = min(int(channels), int(requested))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


def safe_num_heads(embed_dim: int, requested: int) -> int:
    heads = min(int(embed_dim), max(1, int(requested)))
    while heads > 1 and int(embed_dim) % heads != 0:
        heads -= 1
    return heads


class SafeGroupNorm(nn.GroupNorm):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__(safe_num_groups(channels, groups), int(channels), eps=1e-6, affine=True)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SafeGroupNorm(in_channels, groups),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            SafeGroupNorm(out_channels, groups),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class ConditionalResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = SafeGroupNorm(out_channels, groups)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.dropout = nn.Dropout3d(float(dropout))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(F.silu(cond)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return self.conv(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = -math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * scale)
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ZeroConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = zero_module(nn.Conv3d(in_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
