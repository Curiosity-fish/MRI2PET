"""Multi-scale PET fidelity terms used by Exp029."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def local_contrast_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Match local PET contrast while leaving absolute uptake to image losses."""

    kernel_size = max(1, int(kernel_size))
    if kernel_size == 1:
        return F.l1_loss(prediction, target)
    padding = kernel_size // 2
    prediction_mean = F.avg_pool3d(
        prediction, kernel_size, stride=1, padding=padding, count_include_pad=True
    )
    target_mean = F.avg_pool3d(
        target, kernel_size, stride=1, padding=padding, count_include_pad=True
    )
    return F.l1_loss(prediction - prediction_mean, target - target_mean)
