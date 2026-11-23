"""Deterministic low/high PET decomposition (no learnable parameters).

    Y_low_small = AvgPool3d(kernel=4, stride=4)(Y)            # [B,1,40,48,40]
    Y_low       = trilinear_upsample(Y_low_small)             # [B,1,160,192,160]
    Y_res       = Y - Y_low

Numerically Y == Y_low + Y_res holds up to floating-point rounding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def decompose_pet(
    pet: torch.Tensor,
    kernel: int = 4,
    stride: int = 4,
    output_size: Optional[tuple[int, int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (y_low_small, y_low, y_res).

    y_low_small: [B,1,D/k,H/k,W/k] after AvgPool3d
    y_low:       [B,1,D,H,W] after trilinear upsample
    y_res:       [B,1,D,H,W] == y - y_low
    """
    if pet.ndim != 5 or pet.shape[1] != 1:
        raise ValueError(f"Expected PET [B,1,D,H,W], got {tuple(pet.shape)}")
    y_low_small = F.avg_pool3d(pet, kernel_size=int(kernel), stride=int(stride))
    if output_size is None:
        output_size = tuple(int(v) for v in pet.shape[-3:])
    y_low = F.interpolate(
        y_low_small, size=output_size, mode="trilinear", align_corners=False
    )
    y_res = pet - y_low
    return y_low_small, y_low, y_res


def upsample_low_small(
    low_small: torch.Tensor,
    output_size: tuple[int, int, int],
) -> torch.Tensor:
    """Trilinear upsample of a coarse metabolic map to full image resolution."""
    return F.interpolate(
        low_small, size=tuple(int(v) for v in output_size),
        mode="trilinear", align_corners=False,
    )


def low_small_shape(image_size: tuple[int, int, int], kernel: int = 4) -> tuple[int, int, int]:
    return tuple(int(v // kernel) for v in image_size)
