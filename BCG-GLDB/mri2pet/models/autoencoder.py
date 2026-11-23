"""Components used by the MRI2PET models."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn.functional as F
from torch import nn
from mri2pet.models.blocks import Downsample3D, ResidualBlock3D, SafeGroupNorm, Upsample3D


class PatchBackboneEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        base_channels: int,
        norm_groups: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c0, c1, c2 = int(base_channels), int(base_channels) * 2, int(base_channels) * 4
        self.stem = nn.Conv3d(image_channels, c0, 3, padding=1)
        self.level0 = nn.Sequential(
            ResidualBlock3D(c0, c0, norm_groups, dropout),
            ResidualBlock3D(c0, c0, norm_groups, dropout),
        )
        self.down1 = Downsample3D(c0, c1)
        self.level1 = nn.Sequential(
            ResidualBlock3D(c1, c1, norm_groups, dropout),
            ResidualBlock3D(c1, c1, norm_groups, dropout),
        )
        self.down2 = Downsample3D(c1, c2)
        self.level2 = nn.Sequential(
            ResidualBlock3D(c2, c2, norm_groups, dropout),
            ResidualBlock3D(c2, c2, norm_groups, dropout),
        )
        self.out_channels = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.level0(x)
        x = self.down1(x)
        x = self.level1(x)
        x = self.down2(x)
        x = self.level2(x)
        return x


class PatchBackboneDecoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_channels: int,
        base_channels: int,
        norm_groups: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        c0, c1, c2 = int(base_channels), int(base_channels) * 2, int(base_channels) * 4
        self.in_proj = nn.Conv3d(latent_channels, c2, 3, padding=1)
        self.level2 = nn.Sequential(
            ResidualBlock3D(c2, c2, norm_groups, dropout),
            ResidualBlock3D(c2, c2, norm_groups, dropout),
        )
        self.up1 = Upsample3D(c2, c1)
        self.level1 = nn.Sequential(
            ResidualBlock3D(c1, c1, norm_groups, dropout),
            ResidualBlock3D(c1, c1, norm_groups, dropout),
        )
        self.up0 = Upsample3D(c1, c0)
        self.level0 = nn.Sequential(
            ResidualBlock3D(c0, c0, norm_groups, dropout),
            ResidualBlock3D(c0, c0, norm_groups, dropout),
        )
        self.out = nn.Sequential(
            SafeGroupNorm(c0, norm_groups), nn.SiLU(), nn.Conv3d(c0, image_channels, 3, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        x = self.level2(x)
        x = self.up1(x, tuple(int(v) * 2 for v in z.shape[-3:]))
        x = self.level1(x)
        x = self.up0(x, tuple(int(v) * 2 for v in x.shape[-3:]))
        x = self.level0(x)
        return self.out(x)


class SharedPatchAutoencoder(nn.Module):
    downsample_factor = 4

    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 16,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.norm_groups = int(norm_groups)
        self.dropout = float(dropout)
        self.encoder_backbone = PatchBackboneEncoder(
            image_channels, base_channels, norm_groups, dropout
        )
        self.decoder = PatchBackboneDecoder(
            image_channels, latent_channels, base_channels, norm_groups, dropout
        )


class AE(SharedPatchAutoencoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        c2 = self.encoder_backbone.out_channels
        self.latent = nn.Conv3d(c2, self.latent_channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(self.encoder_backbone(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decoder(z), z
