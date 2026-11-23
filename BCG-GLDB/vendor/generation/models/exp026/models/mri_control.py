from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import Downsample3D, ResidualBlock3D, SafeGroupNorm


class MRIControlEncoder3D(nn.Module):
    """MRI feature pyramid aligned with the /4 PET latent resolution."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int, int, int] = (8, 16, 32, 64),
        norm_groups: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        c2, c4, c8, c16 = [int(v) for v in channels]
        self.channels = (c4, c8, c16)
        self.stem = nn.Conv3d(in_channels, c2, kernel_size=3, stride=2, padding=1)
        self.level2 = ResidualBlock3D(c2, c2, norm_groups, dropout)
        self.down4 = Downsample3D(c2, c4)
        self.level4 = nn.Sequential(
            ResidualBlock3D(c4, c4, norm_groups, dropout),
            ResidualBlock3D(c4, c4, norm_groups, dropout),
        )
        self.down8 = Downsample3D(c4, c8)
        self.level8 = nn.Sequential(
            ResidualBlock3D(c8, c8, norm_groups, dropout),
            ResidualBlock3D(c8, c8, norm_groups, dropout),
        )
        self.down16 = Downsample3D(c8, c16)
        self.level16 = nn.Sequential(
            ResidualBlock3D(c16, c16, norm_groups, dropout),
            ResidualBlock3D(c16, c16, norm_groups, dropout),
        )

    def forward(self, mri: torch.Tensor) -> dict[str, torch.Tensor]:
        run = lambda module, value: (
            checkpoint(module, value, use_reentrant=False)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled()
            else module(value)
        )
        x2 = run(self.level2, self.stem(mri))
        high = run(self.level4, self.down4(x2))
        middle = run(self.level8, self.down8(high))
        low = run(self.level16, self.down16(middle))
        return {"high": high, "middle": middle, "low": low}


class MRIPatchPriorHeads3D(nn.Module):
    """Task-aligned heads used to pretrain MRI features against PET patch latents."""

    def __init__(
        self,
        mri_channels: tuple[int, int, int],
        latent_channels: int,
        norm_groups: int = 8,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.heads = nn.ModuleDict(
            {
                level: self._head(channels, latent_channels, norm_groups)
                for level, channels in zip(("high", "middle", "low"), mri_channels)
            }
        )

    @staticmethod
    def _head(in_channels: int, latent_channels: int, norm_groups: int) -> nn.Sequential:
        output = nn.Conv3d(in_channels, latent_channels, kernel_size=1)
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(output.bias)
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            SafeGroupNorm(in_channels, norm_groups),
            nn.SiLU(),
            output,
        )

    def forward(
        self,
        mri_patch_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        predictions = {}
        for level, head in self.heads.items():
            patches = mri_patch_features[level]
            if patches.ndim != 6:
                raise ValueError(f"Expected {level} MRI patches [B,P,C,D,H,W]")
            batch, count = patches.shape[:2]
            flat = patches.reshape(batch * count, *patches.shape[2:])
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                prediction = checkpoint(head, flat, use_reentrant=False)
            else:
                prediction = head(flat)
            predictions[level] = prediction.reshape(batch, count, *prediction.shape[1:])
        return predictions
