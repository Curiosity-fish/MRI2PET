from __future__ import annotations

import torch
from torch import nn

from models.exp025g_patch_latent_vae import PatchLatentDecoder3D, PatchLatentEncoder3D
from models.exp025m_patch_latent_relational_seam_denoiser import (
    Exp025MPatchLatentRelationalSeamDenoiser3D,
)


class Exp025PRegionPatchLatentVAE3D(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        region_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 16,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.region_channels = int(region_channels)
        self.encoder = PatchLatentEncoder3D(
            in_channels=self.image_channels + self.region_channels,
            base_channels=int(base_channels),
            latent_channels=int(latent_channels),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )
        self.decoder = PatchLatentDecoder3D(
            out_channels=self.image_channels,
            base_channels=int(base_channels),
            latent_channels=int(latent_channels),
            norm_groups=int(norm_groups),
            dropout=float(dropout),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(
        self,
        pet: torch.Tensor,
        region_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.region_channels == 0:
            return self.encoder(pet)
        if region_condition is None:
            raise ValueError("region_condition is required when region_channels > 0.")
        return self.encoder(torch.cat([pet, region_condition.to(device=pet.device, dtype=pet.dtype)], dim=1))

    def forward(
        self,
        pet: torch.Tensor,
        region_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(pet, region_condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


class Exp025PRegionMaskPatchLatentDenoiser3D(Exp025MPatchLatentRelationalSeamDenoiser3D):
    def __init__(
        self,
        *args,
        region_channels: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.region_channels = int(region_channels)
        self.region_full_proj = nn.Conv3d(self.region_channels, self.cond_channels, kernel_size=3, padding=1)
        self.region_patch_proj = nn.Conv3d(self.region_channels, self.cond_channels, kernel_size=3, padding=1)

    def encode_condition(
        self,
        mri: torch.Tensor,
        region_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = super().encode_condition(mri)
        if region_full is None:
            return cond
        if tuple(region_full.shape[-3:]) != tuple(cond.shape[-3:]):
            raise ValueError(f"region_full shape {tuple(region_full.shape[-3:])} != latent condition shape {tuple(cond.shape[-3:])}")
        return cond + self.region_full_proj(region_full.to(device=cond.device, dtype=cond.dtype))

    def denoise_patches(
        self,
        x_t_patch: torch.Tensor,
        cond_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        time: torch.Tensor,
        global_context: dict[str, torch.Tensor],
        table_context: dict[str, torch.Tensor] | None = None,
        region_patch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if region_patch is not None:
            cond_patch = cond_patch + self.region_patch_proj(region_patch.to(device=cond_patch.device, dtype=cond_patch.dtype))
        return super().denoise_patches(
            x_t_patch=x_t_patch,
            cond_patch=cond_patch,
            patch_coords=patch_coords,
            time=time,
            global_context=global_context,
            table_context=table_context,
        )
