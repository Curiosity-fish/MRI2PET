from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from models.frequency_guided_regression_unet3d import (
    ConditioningResidualBlock3D,
    Downsample3D,
    FrequencyGuidanceBranch,
    MRIMultiScaleEncoder3D,
    SafeGroupNorm,
    Swish,
    Upsample3D,
)


class CoarseToFineRegressionUNet3D(nn.Module):
    """Task-specific MRI->PET regressor with explicit coarse-to-fine decoding.

    Motivation:
    - PET synthesis needs whole-brain coarse metabolism first.
    - Then local structures can refine region-wise details.

    The network therefore predicts:
    1. a coarse PET field from bottleneck/global features
    2. a residual detail map from decoder features
    Final output = coarse_up + residual
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        num_groups: int = 8,
        frequency_hidden_dim: int = 128,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.channels = [int(base_channels) * int(mult) for mult in channel_mults]
        self.condition_dim = int(condition_dim)

        self.mri_encoder = MRIMultiScaleEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_groups=num_groups,
        )
        self.mri_global_proj = nn.Sequential(
            nn.Linear(self.channels[-1], self.condition_dim),
            Swish(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.frequency_branch = FrequencyGuidanceBranch(
            in_channels=in_channels,
            hidden_dim=int(frequency_hidden_dim),
            out_dim=self.condition_dim,
        )
        self.condition_fuse = nn.Sequential(
            nn.Linear(self.condition_dim * 2, self.condition_dim),
            Swish(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )

        self.input_conv = nn.Conv3d(in_channels, self.channels[0], kernel_size=3, padding=1)
        self.mri_skip_projs = nn.ModuleList([nn.Conv3d(ch, ch, kernel_size=1) for ch in self.channels])

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_ch = self.channels[0]
        for idx, ch in enumerate(self.channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ConditioningResidualBlock3D(current_ch, ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                        ConditioningResidualBlock3D(ch, ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                    ]
                )
            )
            current_ch = ch
            if idx < len(self.channels) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv3d(current_ch, self.channels[idx + 1], kernel_size=3, padding=1),
                        Downsample3D(self.channels[idx + 1]),
                    )
                )
                current_ch = self.channels[idx + 1]

        self.mid1 = ConditioningResidualBlock3D(current_ch, current_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups)
        self.mid2 = ConditioningResidualBlock3D(current_ch, current_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups)

        self.coarse_head = nn.Sequential(
            SafeGroupNorm(current_ch, num_groups=num_groups),
            Swish(),
            nn.Conv3d(current_ch, current_ch // 2, kernel_size=3, padding=1),
            Swish(),
            nn.Conv3d(current_ch // 2, out_channels, kernel_size=1),
        )
        self.coarse_to_decoder = nn.Conv3d(out_channels, self.channels[0], kernel_size=3, padding=1)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(self.channels))
        current_ch = rev_channels[0]
        for skip_ch in rev_channels[1:]:
            self.upsamples.append(
                nn.Sequential(
                    Upsample3D(current_ch),
                    nn.Conv3d(current_ch, skip_ch, kernel_size=3, padding=1),
                )
            )
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ConditioningResidualBlock3D(skip_ch * 2, skip_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                        ConditioningResidualBlock3D(skip_ch, skip_ch, cond_dim=self.condition_dim, dropout=dropout, num_groups=num_groups),
                    ]
                )
            )
            current_ch = skip_ch

        self.residual_head = nn.Sequential(
            SafeGroupNorm(self.channels[0], num_groups=num_groups),
            Swish(),
            nn.Conv3d(self.channels[0], self.channels[0], kernel_size=3, padding=1),
            Swish(),
            nn.Conv3d(self.channels[0], out_channels, kernel_size=3, padding=1),
        )

    def build_condition(self, mri: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        mri_feats, mri_global = self.mri_encoder(mri)
        global_emb = self.mri_global_proj(mri_global)
        freq_emb = self.frequency_branch(mri)
        cond = self.condition_fuse(torch.cat([global_emb, freq_emb], dim=1))
        return mri_feats, cond

    def forward(self, mri: torch.Tensor) -> dict[str, torch.Tensor]:
        mri_feats, cond = self.build_condition(mri)
        x = self.input_conv(mri)
        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            x = x + self.mri_skip_projs[idx](mri_feats[idx])
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        coarse = self.coarse_head(x)
        coarse_up = F.interpolate(coarse, size=mri.shape[-3:], mode="trilinear", align_corners=False)

        skips = list(reversed(skips[:-1]))
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, cond)

        x = x + self.coarse_to_decoder(coarse_up)
        residual = self.residual_head(x)
        prediction = coarse_up + residual
        return {
            "prediction": prediction,
            "coarse_prediction": coarse,
            "coarse_upsampled": coarse_up,
            "residual_prediction": residual,
        }
