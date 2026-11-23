from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import Downsample3D, ResidualBlock3D, SafeGroupNorm, Upsample3D


class PatchEncoder3D(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_channels: int,
        base_channels: int,
        norm_groups: int,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c0, c1, c2 = int(base_channels), int(base_channels) * 2, int(base_channels) * 4
        self.stem = nn.Conv3d(image_channels, c0, kernel_size=3, padding=1)
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
        self.mu = nn.Conv3d(c2, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(c2, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        run = lambda module, value: (
            checkpoint(module, value, use_reentrant=False)
            if self.gradient_checkpointing and torch.is_grad_enabled()
            else module(value)
        )
        x = run(self.level0, self.stem(x))
        x = run(self.level1, self.down1(x))
        x = run(self.level2, self.down2(x))
        return self.mu(x), self.logvar(x).clamp(-30.0, 20.0)


class PatchDecoder3D(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_channels: int,
        base_channels: int,
        norm_groups: int,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c0, c1, c2 = int(base_channels), int(base_channels) * 2, int(base_channels) * 4
        self.in_proj = nn.Conv3d(latent_channels, c2, kernel_size=3, padding=1)
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
            SafeGroupNorm(c0, norm_groups),
            nn.SiLU(),
            nn.Conv3d(c0, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        run = lambda module, value: (
            checkpoint(module, value, use_reentrant=False)
            if self.gradient_checkpointing and torch.is_grad_enabled()
            else module(value)
        )
        x = run(self.level2, self.in_proj(z))
        x = run(self.level1, self.up1(x, tuple(int(v) * 2 for v in z.shape[-3:])))
        x = run(self.level0, self.up0(x, tuple(int(v) * 2 for v in x.shape[-3:])))
        return self.out(x)


class PatchDVAE3D(nn.Module):
    downsample_factor = 4

    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 16,
        norm_groups: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = PatchEncoder3D(
            image_channels,
            latent_channels,
            base_channels,
            norm_groups,
            dropout,
            gradient_checkpointing,
        )
        self.decoder = PatchDecoder3D(
            image_channels,
            latent_channels,
            base_channels,
            norm_groups,
            dropout,
            gradient_checkpointing,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z
