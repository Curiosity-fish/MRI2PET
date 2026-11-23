"""Exp031 HMRD loss primitives (all configurable from the trainer).

Implements the default Exp031 core losses:
  * 3D Laplacian pyramid loss (levels=3, weights 1.0/0.5/0.25)
  * Parc98 ROI loss: SmoothL1 + 0.2 * (1 - PearsonCorr) with numeric stability
  * multi-scale SSIM helper (pooling + SSIM3D)
The Exp030 raw FFT frequency loss is intentionally NOT part of the Exp031
default (frequency_loss.enabled = false) but its implementation is untouched.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from utils.ssim import SSIM3D


def laplacian_pyramid_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    levels: int = 3,
    weights: Sequence[float] = (1.0, 0.5, 0.25),
    final_level_weight: float = 0.25,
) -> torch.Tensor:
    """L1 loss between corresponding 3D Laplacian pyramid levels."""
    levels = max(1, int(levels))
    weights = [float(w) for w in weights]
    if len(weights) < levels:
        weights = list(weights) + [float(weights[-1])] * (levels - len(weights))
    current_prediction = prediction
    current_target = target
    total = prediction.new_zeros(())
    for level in range(levels):
        down_prediction = F.avg_pool3d(current_prediction, 2, 2)
        down_target = F.avg_pool3d(current_target, 2, 2)
        up_prediction = F.interpolate(
            down_prediction, size=current_prediction.shape[-3:],
            mode="trilinear", align_corners=False,
        )
        up_target = F.interpolate(
            down_target, size=current_target.shape[-3:],
            mode="trilinear", align_corners=False,
        )
        total = total + weights[level] * F.l1_loss(
            current_prediction - up_prediction,
            current_target - up_target,
        )
        current_prediction, current_target = down_prediction, down_target
    total = total + float(final_level_weight) * F.l1_loss(
        current_prediction, current_target
    )
    return total


def pearson_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked Pearson correlation over the region dimension (numerically stable)."""
    if valid is None:
        valid = torch.ones_like(prediction)
    valid = valid.bool()
    pred = prediction.float()
    target = target.float()
    if valid.numel() == 0 or not valid.any():
        return prediction.new_zeros(()) + 1.0  # neutral when nothing to compare
    pred_masked = torch.where(valid, pred, pred.new_zeros(()))
    target_masked = torch.where(valid, target, target.new_zeros(()))
    count = valid.float().sum().clamp_min(1)
    mean_pred = (pred_masked * valid.float()).sum() / count
    mean_target = (target_masked * valid.float()).sum() / count
    centered_pred = torch.where(valid, pred - mean_pred, pred.new_zeros(()))
    centered_target = torch.where(valid, target - mean_target, target.new_zeros(()))
    covariance = (centered_pred * centered_target * valid.float()).sum()
    std_pred = ((centered_pred.square() * valid.float()).sum()).sqrt().clamp_min(eps)
    std_target = ((centered_target.square() * valid.float()).sum()).sqrt().clamp_min(eps)
    return covariance / (std_pred * std_target).clamp_min(eps)


def roi_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    corr_weight: float = 0.2,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L_roi = SmoothL1(pred, gt) + corr_weight * (1 - PearsonCorr)."""
    if valid is None:
        valid = torch.ones_like(prediction)
    valid_float = valid.float()
    if valid_float.sum() > 0:
        value_loss = F.smooth_l1_loss(
            prediction * valid_float, target * valid_float, reduction="sum"
        ) / valid_float.sum().clamp_min(1e-6)
    else:
        value_loss = prediction.new_zeros(()) + F.smooth_l1_loss(prediction, target)
    corr = pearson_correlation(prediction, target, valid=valid, eps=eps)
    corr_loss = (1.0 - corr).clamp_min(0.0)
    loss = value_loss + float(corr_weight) * corr_loss
    metrics = {
        "roi_value": value_loss.detach(),
        "roi_corr": corr.detach(),
        "roi_corr_loss": corr_loss.detach(),
    }
    return loss, metrics


def multiscale_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim: SSIM3D,
    factor: int = 2,
    levels: int = 1,
) -> torch.Tensor:
    """1 - SSIM on progressively pooled volumes."""
    total = prediction.new_zeros(())
    current_prediction = prediction
    current_target = target
    for _ in range(max(1, int(levels))):
        total = total + (1.0 - ssim(current_prediction, current_target))
        current_prediction = F.avg_pool3d(current_prediction, int(factor), int(factor))
        current_target = F.avg_pool3d(current_target, int(factor), int(factor))
    return total


__all__ = [
    "laplacian_pyramid_loss",
    "pearson_correlation",
    "roi_loss",
    "multiscale_ssim_loss",
]
