from __future__ import annotations

import torch
from torch import nn

from models.exp025d_global_region_patch_denoiser import (
    ConditionalResBlock3D,
    Downsample3D,
    ResidualConvBlock3D,
)
from models.exp025p_region_patch_latent import Exp025PRegionMaskPatchLatentDenoiser3D
from models.global_local_mamba_patch_denoiser import AxialBiMambaBlock


class MambaFullLatentGlobalEncoder3D(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        cond_channels: int,
        channels: tuple[int, int] = (32, 64),
        cond_dim: int = 256,
        norm_groups: int = 8,
        dropout: float = 0.0,
        mamba_depth: int = 1,
        mamba_d_state: int = 4,
        mamba_d_conv: int = 3,
        mamba_expand: int = 1,
        mamba_axes: tuple[str, ...] = ("x", "y", "z"),
        mamba_bidirectional: bool = False,
        mamba_line_chunk_size: int = 512,
        mamba_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        c40, c20 = [int(v) for v in channels]
        self.stem = nn.Conv3d(int(latent_channels) + int(cond_channels), c40, kernel_size=3, padding=1)
        self.pre = ResidualConvBlock3D(c40, c40, norm_groups, dropout)
        self.time_block = ConditionalResBlock3D(c40, c40, cond_dim, norm_groups, dropout)
        self.mamba_blocks = nn.ModuleList(
            [
                AxialBiMambaBlock(
                    dim=c40,
                    condition_dim=int(cond_dim),
                    d_state=int(mamba_d_state),
                    d_conv=int(mamba_d_conv),
                    expand=int(mamba_expand),
                    bidirectional=bool(mamba_bidirectional),
                    active_axes=tuple(mamba_axes),
                    line_chunk_size=int(mamba_line_chunk_size),
                    gradient_checkpointing=bool(mamba_gradient_checkpointing),
                )
                for _ in range(int(mamba_depth))
            ]
        )
        self.down20 = Downsample3D(c40, c20)
        self.block20 = ConditionalResBlock3D(c20, c20, cond_dim, norm_groups, dropout)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x_t_full: torch.Tensor, cond_full: torch.Tensor, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        x40 = self.pre(self.stem(torch.cat([x_t_full, cond_full], dim=1)))
        x40 = self.time_block(x40, cond)
        table_condition = torch.zeros_like(cond)
        table_reliability = x40.new_zeros((x40.shape[0], x40.shape[1]))
        for block in self.mamba_blocks:
            x40 = block(
                x40,
                base_condition=cond,
                table_condition=table_condition,
                table_reliability=table_reliability,
            )
        x20 = self.block20(self.down20(x40), cond)
        return {"40": x40, "20": x20, "pooled": self.pool(x20).flatten(1)}


class Exp025QGlobalMambaLocalUNetPatchLatentDenoiser3D(Exp025PRegionMaskPatchLatentDenoiser3D):
    def __init__(
        self,
        *args,
        global_mamba_depth: int = 1,
        global_mamba_d_state: int = 4,
        global_mamba_d_conv: int = 3,
        global_mamba_expand: int = 1,
        global_mamba_axes: tuple[str, ...] = ("x", "y", "z"),
        global_mamba_bidirectional: bool = False,
        global_mamba_line_chunk_size: int = 512,
        global_mamba_gradient_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.global_encoder = MambaFullLatentGlobalEncoder3D(
            latent_channels=self.latent_channels,
            cond_channels=self.cond_channels,
            channels=tuple(int(v) for v in kwargs.get("global_channels", (32, 64))),
            cond_dim=int(kwargs.get("time_dim", 256)),
            norm_groups=int(kwargs.get("norm_groups", 8)),
            dropout=float(kwargs.get("dropout", 0.0)),
            mamba_depth=int(global_mamba_depth),
            mamba_d_state=int(global_mamba_d_state),
            mamba_d_conv=int(global_mamba_d_conv),
            mamba_expand=int(global_mamba_expand),
            mamba_axes=tuple(global_mamba_axes),
            mamba_bidirectional=bool(global_mamba_bidirectional),
            mamba_line_chunk_size=int(global_mamba_line_chunk_size),
            mamba_gradient_checkpointing=bool(global_mamba_gradient_checkpointing),
        )
