"""HMRD local denoiser: the stable patch-wise 3D U-Net plus the new HMRD
conditions.

Keeps the exact Exp026 LocalControlDenoiser3D backbone (patch-wise 3D U-Net,
MRI zero-conv injections, table prior zero-conv injections, global Mamba
FiLM fusions) and adds three condition paths required by Exp031:

  * coarse PET latent pyramid injection (zero-init convs)
  * Parc98 region-token FiLM fusions (zero-init)
  * region-token additive time-condition (zero-init linear)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from exp026.models.blocks import ZeroConv3D, zero_module
from exp026.models.local_denoiser import (
    GlobalFeatureFusion,
    LocalControlDenoiser3D,
)


class HMRDLocalControlDenoiser3D(LocalControlDenoiser3D):
    def __init__(
        self,
        latent_channels: int,
        mri_channels: tuple[int, int, int],
        local_channels: tuple[int, int, int] = (32, 64, 96),
        time_dim: int = 128,
        table_token_dim: int = 128,
        global_token_dim: int = 128,
        region_token_dim: int = 128,
        grid_size: tuple[int, int, int] = (4, 4, 4),
        norm_groups: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            latent_channels=latent_channels,
            mri_channels=mri_channels,
            local_channels=local_channels,
            time_dim=time_dim,
            table_token_dim=table_token_dim,
            global_token_dim=global_token_dim,
            grid_size=grid_size,
            norm_groups=norm_groups,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        c0, c1, c2 = [int(v) for v in local_channels]
        self.region_token_dim = int(region_token_dim)
        self.region_to_condition = zero_module(
            nn.Linear(self.region_token_dim, time_dim)
        )
        self.region_fusions = nn.ModuleDict(
            {
                "high_encoder": GlobalFeatureFusion(self.region_token_dim, c0),
                "middle_encoder": GlobalFeatureFusion(self.region_token_dim, c1),
                "low": GlobalFeatureFusion(self.region_token_dim, c2),
                "middle_decoder": GlobalFeatureFusion(self.region_token_dim, c1),
                "high_decoder": GlobalFeatureFusion(self.region_token_dim, c0),
            }
        )
        self.coarse_injections = nn.ModuleDict(
            {
                "high_encoder": ZeroConv3D(latent_channels, c0),
                "middle_encoder": ZeroConv3D(latent_channels, c1),
                "low": ZeroConv3D(latent_channels, c2),
                "middle_decoder": ZeroConv3D(latent_channels, c1),
                "high_decoder": ZeroConv3D(latent_channels, c0),
            }
        )

    def _condition(
        self,
        timesteps: torch.Tensor,
        patch_coords: torch.Tensor,
        subject_token: Optional[torch.Tensor],
        global_token: Optional[torch.Tensor],
        region_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = super()._condition(timesteps, patch_coords, subject_token, global_token)
        if region_token is not None:
            cond = cond + self.region_to_condition(region_token).to(dtype=cond.dtype)
        return cond

    def _inject(
        self,
        x: torch.Tensor,
        level: str,
        mri_features: dict[str, torch.Tensor],
        table_prior: Optional[dict[str, torch.Tensor]],
        global_token: Optional[torch.Tensor],
        source_level: str,
        coarse_prior: Optional[dict[str, torch.Tensor]] = None,
        region_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = super()._inject(
            x, level, mri_features, table_prior, global_token, source_level
        )
        if coarse_prior is not None:
            x = self._add_control(x, coarse_prior[source_level], self.coarse_injections[level])
        if region_token is not None:
            x = self.region_fusions[level](x, region_token)
        return x

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        patch_coords: torch.Tensor,
        mri_features: dict[str, torch.Tensor],
        table_prior: Optional[dict[str, torch.Tensor]] = None,
        subject_token: Optional[torch.Tensor] = None,
        global_token: Optional[torch.Tensor] = None,
        coarse_prior: Optional[dict[str, torch.Tensor]] = None,
        region_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._condition(
            timesteps, patch_coords, subject_token, global_token, region_token
        )
        x0 = self._run_block(self.enc0a, self.stem(noisy_latent), cond)
        x0 = self._run_block(self.enc0b, x0, cond)
        x0 = self._inject(
            x0, "high_encoder", mri_features, table_prior, global_token,
            "high", coarse_prior=coarse_prior, region_token=region_token,
        )
        skip0 = x0

        x1 = self._run_block(self.enc1a, self.down1(x0), cond)
        x1 = self._run_block(self.enc1b, x1, cond)
        x1 = self._inject(
            x1, "middle_encoder", mri_features, table_prior, global_token,
            "middle", coarse_prior=coarse_prior, region_token=region_token,
        )
        skip1 = x1

        x2 = self._run_block(self.mid1, self.down2(x1), cond)
        x2 = self._run_block(self.mid2, x2, cond)
        x2 = self._inject(
            x2, "low", mri_features, table_prior, global_token,
            "low", coarse_prior=coarse_prior, region_token=region_token,
        )

        x = self.up1(x2, tuple(skip1.shape[-3:]))
        x = self._run_block(self.dec1a, torch.cat([x, skip1], dim=1), cond)
        x = self._run_block(self.dec1b, x, cond)
        x = self._inject(
            x, "middle_decoder", mri_features, table_prior, global_token,
            "middle", coarse_prior=coarse_prior, region_token=region_token,
        )
        x = self.up0(x, tuple(skip0.shape[-3:]))
        x = self._run_block(self.dec0a, torch.cat([x, skip0], dim=1), cond)
        x = self._run_block(self.dec0b, x, cond)
        x = self._inject(
            x, "high_decoder", mri_features, table_prior, global_token,
            "high", coarse_prior=coarse_prior, region_token=region_token,
        )
        return self.out(x)


__all__ = ["HMRDLocalControlDenoiser3D"]
