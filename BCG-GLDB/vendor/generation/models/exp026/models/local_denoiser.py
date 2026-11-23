from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    ConditionalResidualBlock3D,
    Downsample3D,
    SafeGroupNorm,
    SinusoidalTimeEmbedding,
    Upsample3D,
    ZeroConv3D,
    zero_module,
)


class GlobalFeatureFusion(nn.Module):
    def __init__(self, token_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = zero_module(nn.Linear(token_dim, channels * 2))

    def forward(self, x: torch.Tensor, token: Optional[torch.Tensor]) -> torch.Tensor:
        if token is None:
            return x
        scale, shift = self.affine(token).chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]


class LocalControlDenoiser3D(nn.Module):
    """Patch-wise 3D U-Net. MRI is structural control; global Mamba is auxiliary."""

    def __init__(
        self,
        latent_channels: int,
        mri_channels: tuple[int, int, int],
        local_channels: tuple[int, int, int] = (32, 64, 96),
        time_dim: int = 128,
        table_token_dim: int = 128,
        global_token_dim: int = 128,
        grid_size: tuple[int, int, int] = (4, 4, 4),
        norm_groups: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c0, c1, c2 = [int(v) for v in local_channels]
        self.grid_size = tuple(int(v) for v in grid_size)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(3, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.subject_to_condition = nn.Linear(table_token_dim, time_dim)
        self.global_to_condition = zero_module(nn.Linear(global_token_dim, time_dim))

        self.stem = nn.Conv3d(latent_channels, c0, kernel_size=3, padding=1)
        self.enc0a = ConditionalResidualBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.enc0b = ConditionalResidualBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.down1 = Downsample3D(c0, c1)
        self.enc1a = ConditionalResidualBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.enc1b = ConditionalResidualBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.down2 = Downsample3D(c1, c2)
        self.mid1 = ConditionalResidualBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.mid2 = ConditionalResidualBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.up1 = Upsample3D(c2, c1)
        self.dec1a = ConditionalResidualBlock3D(c1 + c1, c1, time_dim, norm_groups, dropout)
        self.dec1b = ConditionalResidualBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.up0 = Upsample3D(c1, c0)
        self.dec0a = ConditionalResidualBlock3D(c0 + c0, c0, time_dim, norm_groups, dropout)
        self.dec0b = ConditionalResidualBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.out = nn.Sequential(
            SafeGroupNorm(c0, norm_groups),
            nn.SiLU(),
            nn.Conv3d(c0, latent_channels, kernel_size=3, padding=1),
        )

        self.mri_injections = nn.ModuleDict(
            {
                "high_encoder": ZeroConv3D(mri_channels[0], c0),
                "middle_encoder": ZeroConv3D(mri_channels[1], c1),
                "low": ZeroConv3D(mri_channels[2], c2),
                "middle_decoder": ZeroConv3D(mri_channels[1], c1),
                "high_decoder": ZeroConv3D(mri_channels[0], c0),
            }
        )
        self.table_injections = nn.ModuleDict(
            {
                "high_encoder": ZeroConv3D(latent_channels, c0),
                "middle_encoder": ZeroConv3D(latent_channels, c1),
                "low": ZeroConv3D(latent_channels, c2),
                "middle_decoder": ZeroConv3D(latent_channels, c1),
                "high_decoder": ZeroConv3D(latent_channels, c0),
            }
        )
        self.global_fusions = nn.ModuleDict(
            {
                "high_encoder": GlobalFeatureFusion(global_token_dim, c0),
                "middle_encoder": GlobalFeatureFusion(global_token_dim, c1),
                "low": GlobalFeatureFusion(global_token_dim, c2),
                "middle_decoder": GlobalFeatureFusion(global_token_dim, c1),
                "high_decoder": GlobalFeatureFusion(global_token_dim, c0),
            }
        )

    def embed_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embedding(timesteps)

    def _run_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, cond, use_reentrant=False)
        return block(x, cond)

    def _condition(
        self,
        timesteps: torch.Tensor,
        patch_coords: torch.Tensor,
        subject_token: Optional[torch.Tensor],
        global_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        denominators = patch_coords.new_tensor(
            [max(v - 1, 1) for v in self.grid_size], dtype=torch.float32
        )
        position = patch_coords.float() / denominators
        cond = self.embed_time(timesteps) + self.position_embedding(position)
        if subject_token is not None:
            cond = cond + self.subject_to_condition(subject_token)
        if global_token is not None:
            cond = cond + self.global_to_condition(global_token)
        return cond

    @staticmethod
    def _add_control(x: torch.Tensor, control: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if tuple(control.shape[-3:]) != tuple(x.shape[-3:]):
            control = F.interpolate(
                control, size=x.shape[-3:], mode="trilinear", align_corners=False
            )
        return x + projection(control).to(dtype=x.dtype)

    def _inject(
        self,
        x: torch.Tensor,
        level: str,
        mri_features: dict[str, torch.Tensor],
        table_prior: Optional[dict[str, torch.Tensor]],
        global_token: Optional[torch.Tensor],
        source_level: str,
    ) -> torch.Tensor:
        x = self._add_control(x, mri_features[source_level], self.mri_injections[level])
        if table_prior is not None:
            x = self._add_control(x, table_prior[source_level], self.table_injections[level])
        return self.global_fusions[level](x, global_token)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        patch_coords: torch.Tensor,
        mri_features: dict[str, torch.Tensor],
        table_prior: Optional[dict[str, torch.Tensor]] = None,
        subject_token: Optional[torch.Tensor] = None,
        global_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._condition(timesteps, patch_coords, subject_token, global_token)
        x0 = self._run_block(self.enc0a, self.stem(noisy_latent), cond)
        x0 = self._run_block(self.enc0b, x0, cond)
        x0 = self._inject(
            x0, "high_encoder", mri_features, table_prior, global_token, "high"
        )
        skip0 = x0

        x1 = self._run_block(self.enc1a, self.down1(x0), cond)
        x1 = self._run_block(self.enc1b, x1, cond)
        x1 = self._inject(
            x1, "middle_encoder", mri_features, table_prior, global_token, "middle"
        )
        skip1 = x1

        x2 = self._run_block(self.mid1, self.down2(x1), cond)
        x2 = self._run_block(self.mid2, x2, cond)
        x2 = self._inject(x2, "low", mri_features, table_prior, global_token, "low")

        x = self.up1(x2, tuple(skip1.shape[-3:]))
        x = self._run_block(self.dec1a, torch.cat([x, skip1], dim=1), cond)
        x = self._run_block(self.dec1b, x, cond)
        x = self._inject(
            x, "middle_decoder", mri_features, table_prior, global_token, "middle"
        )
        x = self.up0(x, tuple(skip0.shape[-3:]))
        x = self._run_block(self.dec0a, torch.cat([x, skip0], dim=1), cond)
        x = self._run_block(self.dec0b, x, cond)
        x = self._inject(
            x, "high_decoder", mri_features, table_prior, global_token, "high"
        )
        return self.out(x)
