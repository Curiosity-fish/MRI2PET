"""GlobalMetabolicPrior3D.

Inputs:  MRI multi-scale control features (high/middle/low from MRIControlEncoder3D)
         + missing-aware Table tokens (from JointTableMRIPrior3D).
Outputs: coarse_low_small [B,1,40,48,40]  (the AvgPool4 target of GT PET)
         region_uptake    [B,98]           (per-Parc98-ROI mean uptake)
         coarse_pet       [B,1,160,192,160] (trilinear upsample of coarse_low_small)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from exp026.models.blocks import SafeGroupNorm
from exp031.models.decomposition import upsample_low_small


class GlobalMetabolicPrior3D(nn.Module):
    def __init__(
        self,
        mri_channels: tuple[int, int, int] = (16, 32, 64),
        fuse_channels: int = 64,
        region_token_dim: int = 128,
        num_regions: int = 98,
        table_token_dim: int = 128,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        self.table_token_dim = int(table_token_dim)
        c0, c1, c2 = [int(v) for v in mri_channels]
        self.fuse = nn.Sequential(
            nn.Conv3d(c0 + c1 + c2, fuse_channels, kernel_size=3, padding=1),
            SafeGroupNorm(fuse_channels, norm_groups),
            nn.SiLU(),
            nn.Conv3d(fuse_channels, fuse_channels, kernel_size=3, padding=1),
            SafeGroupNorm(fuse_channels, norm_groups),
            nn.SiLU(),
        )
        self.coarse_head = nn.Sequential(
            nn.Conv3d(fuse_channels, fuse_channels // 2, kernel_size=3, padding=1),
            SafeGroupNorm(fuse_channels // 2, norm_groups),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(fuse_channels // 2, 1, kernel_size=1),
        )
        # Region condition = global MRI feature [fuse_channels] concat
        # table summary [table_token_dim] (subject token added elementwise).
        self.region_head = nn.Sequential(
            nn.Linear(fuse_channels + table_token_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.num_regions),
        )
        # Small weight init so the first coarse/region outputs are near zero
        # (zero-init correction convention used elsewhere in this repo).
        for module in self.coarse_head.modules():
            if isinstance(module, nn.Conv3d) and module.out_channels == 1:
                nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(module.bias)
        for module in self.region_head.modules():
            if isinstance(module, nn.Linear) and module.out_features == self.num_regions:
                nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        controls: dict[str, torch.Tensor],
        table_tokens: Optional[torch.Tensor] = None,
        subject_token: Optional[torch.Tensor] = None,
        output_size: Optional[tuple[int, int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        high = controls["high"]
        middle = controls["middle"]
        low = controls["low"]
        target_size = tuple(int(v) for v in high.shape[-3:])
        middle = F.interpolate(
            middle, size=target_size, mode="trilinear", align_corners=False
        )
        low = F.interpolate(low, size=target_size, mode="trilinear", align_corners=False)
        fused = self.fuse(torch.cat([high, middle, low], dim=1))
        coarse_low_small = self.coarse_head(fused)  # [B,1,40,48,40]

        global_feature = fused.mean(dim=(-3, -2, -1))  # [B, fuse_channels]
        if table_tokens is not None:
            table_summary = table_tokens.mean(dim=1)  # [B, table_token_dim]
            if subject_token is not None:
                table_summary = table_summary + subject_token
        elif subject_token is not None:
            table_summary = subject_token
        else:
            table_summary = torch.zeros(
                (global_feature.shape[0], self.table_token_dim),
                device=global_feature.device,
                dtype=global_feature.dtype,
            )
        condition = torch.cat([global_feature, table_summary], dim=1)
        region_uptake = self.region_head(condition)  # [B,98]

        if output_size is None:
            output_size = tuple(int(v) * 4 for v in target_size)
        coarse_pet = upsample_low_small(coarse_low_small, output_size)
        return {
            "coarse_low_small": coarse_low_small,
            "region_uptake": region_uptake,
            "coarse_pet": coarse_pet,
        }


__all__ = ["GlobalMetabolicPrior3D"]
