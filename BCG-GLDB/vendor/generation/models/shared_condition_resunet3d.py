from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from models.residual_bridge_resunet3d import (
    ResidualBridgeResUNet3D,
)


class SharedConditionResUNet3D(ResidualBridgeResUNet3D):
    """One denoiser body for both full-volume latent bridge and region-guided patch bridge."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 3,
        num_heads: int = 4,
        use_region_mask: bool = False,
        use_patch_position: bool = False,
    ) -> None:
        self.use_region_mask = bool(use_region_mask)
        self.use_patch_position = bool(use_patch_position)
        extra_in = int(self.use_region_mask) + (3 if self.use_patch_position else 0)
        super().__init__(
            in_channels=int(in_channels) + extra_in,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_dim=time_dim,
            dropout=dropout,
            num_groups=num_groups,
            use_attention=use_attention,
            attention_level=attention_level,
            num_heads=num_heads,
        )

    def _make_position_channels(
        self,
        patch_pos: Optional[torch.Tensor],
        spatial_size: tuple[int, int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if patch_pos is None:
            return torch.zeros((batch_size, 3, *spatial_size), device=device, dtype=dtype)
        if patch_pos.ndim != 2 or patch_pos.shape[1] != 3:
            raise ValueError(f"patch_pos must be [B,3], got {tuple(patch_pos.shape)}")
        pos = patch_pos.to(device=device, dtype=dtype)
        return pos[:, :, None, None, None].expand(batch_size, 3, *spatial_size)

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        patch_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [x_t, condition]
        if self.use_region_mask:
            if region_mask is None:
                region_mask = torch.zeros(
                    (x_t.shape[0], 1, *x_t.shape[2:]),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
            if region_mask.shape[2:] != x_t.shape[2:]:
                region_mask = F.interpolate(region_mask.float(), size=x_t.shape[2:], mode="nearest")
            parts.append(region_mask.to(device=x_t.device, dtype=x_t.dtype))
        if self.use_patch_position:
            parts.append(
                self._make_position_channels(
                    patch_pos,
                    tuple(x_t.shape[2:]),
                    int(x_t.shape[0]),
                    x_t.device,
                    x_t.dtype,
                )
            )
        x = torch.cat(parts, dim=1)
        return super().forward(x[:, :1], x[:, 1:], t)
