from __future__ import annotations

from typing import Any

import torch


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _corr(left: torch.Tensor, right: torch.Tensor, spearman: bool = False) -> float:
    if left.numel() < 2:
        return float("nan")
    if spearman:
        left, right = _rank(left), _rank(right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum().div(denominator.clamp_min(1e-8)).item())


def parc98_roi_metrics(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = prediction.float()
    true = target.float()
    label = labels.long()
    pred_means, true_means = [], []
    abs_errors, sq_errors = [], []
    for region_id in range(1, 99):
        region = label[0, 0] == region_id
        if not region.any():
            continue
        p = pred[0, 0][region]
        t = true[0, 0][region]
        pred_means.append(p.mean())
        true_means.append(t.mean())
        abs_errors.append((p - t).abs().mean())
        sq_errors.append((p - t).square().mean())
    if not abs_errors:
        return {"roi_mae": float("nan"), "roi_rmse": float("nan"), "roi_pearson": float("nan"), "roi_spearman": float("nan")}
    pred_means_t = torch.stack(pred_means)
    true_means_t = torch.stack(true_means)
    return {
        "roi_mae": float(torch.stack(abs_errors).mean().item()),
        "roi_rmse": float(torch.stack(sq_errors).mean().sqrt().item()),
        "roi_pearson": _corr(pred_means_t, true_means_t, spearman=False),
        "roi_spearman": _corr(pred_means_t, true_means_t, spearman=True),
    }


def patch_boundary_metrics(prediction: torch.Tensor, target: torch.Tensor, patch_size: tuple[int, int, int]) -> dict[str, float]:
    pred = prediction.float()
    true = target.float()
    value_errors = []
    gradient_errors = []
    for dimension, width in enumerate(patch_size):
        axis = 2 + dimension
        for boundary in range(int(width), pred.shape[axis], int(width)):
            left_index = boundary - 1
            right_index = boundary
            if left_index < 0 or right_index >= pred.shape[axis]:
                continue
            left = [slice(None)] * pred.ndim
            right = [slice(None)] * pred.ndim
            left[axis] = left_index
            right[axis] = right_index
            value_errors.append((pred[tuple(left)] - true[tuple(left)]).abs())
            value_errors.append((pred[tuple(right)] - true[tuple(right)]).abs())
            pred_jump = pred[tuple(right)] - pred[tuple(left)]
            true_jump = true[tuple(right)] - true[tuple(left)]
            gradient_errors.append((pred_jump - true_jump).abs())
    if not value_errors:
        return {"boundary_mae": float("nan"), "boundary_gradient_error": float("nan")}
    return {
        "boundary_mae": float(torch.cat([v.reshape(-1) for v in value_errors]).mean().item()),
        "boundary_gradient_error": float(torch.cat([v.reshape(-1) for v in gradient_errors]).mean().item()) if gradient_errors else float("nan"),
    }

