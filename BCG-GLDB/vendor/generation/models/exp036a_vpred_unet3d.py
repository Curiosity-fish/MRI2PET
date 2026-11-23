from __future__ import annotations

import torch
from torch import nn

from models.exp034a_basic_unet3d import BasicUNet3D


class Exp036AVPredUNet3D(BasicUNet3D):
    """Conditional latent U-Net: noisy PET latent + MRI latent -> v prediction."""

    def __init__(self, latent_channels: int = 4, **kwargs) -> None:
        super().__init__(in_channels=latent_channels * 2,
                         out_channels=latent_channels, **kwargs)
        self.latent_channels = latent_channels

    def forward(self, x_t: torch.Tensor, mri_latent: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        if x_t.shape != mri_latent.shape:
            raise ValueError(f"latent shapes differ: {tuple(x_t.shape)} vs {tuple(mri_latent.shape)}")
        return super().forward(torch.cat([x_t, mri_latent], dim=1), t)
