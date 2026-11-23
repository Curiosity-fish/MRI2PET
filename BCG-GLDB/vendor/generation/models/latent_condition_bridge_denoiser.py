from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from models.shared_condition_resunet3d import SharedConditionResUNet3D


class LatentConditionBridgeDenoiser3D(nn.Module):
    """PET latent bridge denoiser with a separate MRI encoder."""

    def __init__(
        self,
        latent_channels: int = 1,
        cond_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 2,
        num_heads: int = 4,
        mri_encoder_channels: Iterable[int] = (8, 16, 32),
    ) -> None:
        super().__init__()
        enc_channels = [1] + [int(v) for v in mri_encoder_channels]
        blocks = []
        for idx in range(len(enc_channels) - 1):
            stride = 2 if idx < len(enc_channels) - 2 else 1
            blocks.extend(
                [
                    nn.Conv3d(enc_channels[idx], enc_channels[idx + 1], kernel_size=3, stride=stride, padding=1),
                    nn.GroupNorm(num_groups=1, num_channels=enc_channels[idx + 1]),
                    nn.SiLU(),
                ]
            )
        self.mri_encoder = nn.Sequential(*blocks)
        self.mri_proj = nn.Conv3d(enc_channels[-1], int(cond_channels), kernel_size=1)
        self.denoiser = SharedConditionResUNet3D(
            in_channels=int(latent_channels) + int(cond_channels),
            out_channels=int(latent_channels),
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_dim=time_dim,
            dropout=dropout,
            num_groups=num_groups,
            use_attention=use_attention,
            attention_level=attention_level,
            num_heads=num_heads,
            use_region_mask=False,
            use_patch_position=False,
        )

    def encode_condition(self, mri: torch.Tensor, out_size: tuple[int, int, int]) -> torch.Tensor:
        feature = self.mri_encoder(mri)
        if tuple(feature.shape[-3:]) != tuple(out_size):
            feature = F.interpolate(feature, size=out_size, mode="trilinear", align_corners=False)
        return self.mri_proj(feature)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tuple(condition.shape[-3:]) == tuple(x_t.shape[-3:]):
            cond = condition
        else:
            cond = self.encode_condition(condition, tuple(x_t.shape[-3:]))
        return self.denoiser(x_t, cond, t)
