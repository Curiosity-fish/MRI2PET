"""Small, domain-aligned losses for MRI-to-PET structural fidelity."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _edge_weight(mri_difference: torch.Tensor, strength: float) -> torch.Tensor:
    """Bound MRI edge emphasis to avoid letting MRI dominate PET intensity."""

    flat = mri_difference.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1, 1)
    std = flat.std(dim=1, unbiased=False).view(-1, 1, 1, 1, 1).clamp_min(1e-4)
    normalized = (mri_difference - mean) / std
    return 1.0 + float(strength) * torch.sigmoid(normalized)


def mri_edge_guided_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mri: torch.Tensor,
    mask: torch.Tensor,
    *,
    edge_strength: float = 0.35,
) -> torch.Tensor:
    """Match PET gradients, emphasizing reliable MRI anatomical transitions.

    MRI is used only as a bounded weighting field.  The target PET gradient is
    still the supervision signal, so this does not force PET to copy MRI.
    """

    terms = []
    for dimension in (-3, -2, -1):
        prediction_difference = torch.diff(prediction, dim=dimension)
        target_difference = torch.diff(target, dim=dimension)
        mri_difference = torch.diff(mri, dim=dimension).abs()
        weight = _edge_weight(mri_difference, edge_strength)
        left = [slice(None)] * mask.ndim
        right = [slice(None)] * mask.ndim
        left[dimension] = slice(1, None)
        right[dimension] = slice(None, -1)
        valid = mask[tuple(left)] * mask[tuple(right)]
        weighted_valid = valid * weight
        terms.append(
            (prediction_difference - target_difference).abs().mul(weighted_valid).sum()
            / weighted_valid.sum().clamp_min(1e-6)
        )
    return torch.stack(terms).mean()


def low_frequency_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    factor: int = 2,
) -> torch.Tensor:
    """Preserve coarse PET uptake without adding a second full-volume decoder."""

    factor = max(1, int(factor))
    if factor == 1:
        return F.l1_loss(prediction, target)
    pooled_prediction = F.avg_pool3d(prediction, factor, factor)
    pooled_target = F.avg_pool3d(target, factor, factor)
    return F.l1_loss(pooled_prediction, pooled_target)
