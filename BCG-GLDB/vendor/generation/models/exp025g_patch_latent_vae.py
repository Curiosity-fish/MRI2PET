from __future__ import annotations

import torch
from torch import nn

from models.exp025d_global_region_patch_denoiser import (
    Downsample3D,
    ResidualConvBlock3D,
    SafeGroupNorm,
    Upsample3D,
)


class PatchLatentEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        latent_channels: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c40 = int(base_channels)
        c20 = int(base_channels) * 2
        c10 = int(base_channels) * 4
        self.stem = nn.Conv3d(int(in_channels), c40, kernel_size=3, padding=1)
        self.block40a = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.block40b = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.down20 = Downsample3D(c40, c20)
        self.block20a = ResidualConvBlock3D(c20, c20, norm_groups, dropout)
        self.block20b = ResidualConvBlock3D(c20, c20, norm_groups, dropout)
        self.down10 = Downsample3D(c20, c10)
        self.block10a = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.block10b = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.mu = nn.Conv3d(c10, int(latent_channels), kernel_size=1)
        self.logvar = nn.Conv3d(c10, int(latent_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.block40b(self.block40a(x))
        x = self.down20(x)
        x = self.block20b(self.block20a(x))
        x = self.down10(x)
        x = self.block10b(self.block10a(x))
        return self.mu(x), self.logvar(x)


class PatchLatentDecoder3D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        latent_channels: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c40 = int(base_channels)
        c20 = int(base_channels) * 2
        c10 = int(base_channels) * 4
        self.in_proj = nn.Conv3d(int(latent_channels), c10, kernel_size=3, padding=1)
        self.block10a = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.block10b = ResidualConvBlock3D(c10, c10, norm_groups, dropout)
        self.up20 = Upsample3D(c10, c20)
        self.block20a = ResidualConvBlock3D(c20, c20, norm_groups, dropout)
        self.block20b = ResidualConvBlock3D(c20, c20, norm_groups, dropout)
        self.up40 = Upsample3D(c20, c40)
        self.block40a = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.block40b = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.out = nn.Sequential(
            SafeGroupNorm(c40, norm_groups),
            nn.SiLU(),
            nn.Conv3d(c40, int(out_channels), kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.block10b(self.block10a(self.in_proj(z)))
        x = self.up20(x, (z.shape[-3] * 2, z.shape[-2] * 2, z.shape[-1] * 2))
        x = self.block20b(self.block20a(x))
        x = self.up40(x, (x.shape[-3] * 2, x.shape[-2] * 2, x.shape[-1] * 2))
        x = self.block40b(self.block40a(x))
        return self.out(x)


class Exp025GPatchLatentVAE3D(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 16,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = PatchLatentEncoder3D(
            in_channels=int(image_channels),
            base_channels=int(base_channels),
            latent_channels=int(latent_channels),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )
        self.decoder = PatchLatentDecoder3D(
            out_channels=int(image_channels),
            base_channels=int(base_channels),
            latent_channels=int(latent_channels),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
