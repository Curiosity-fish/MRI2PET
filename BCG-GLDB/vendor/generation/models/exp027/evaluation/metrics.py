"""Full image, region and patch-boundary metrics for Exp027 evaluation."""

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


def parc98_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    brain_mask: torch.Tensor,
    graph: Any,
) -> dict[str, float]:
    """Compute sample-averaged Parc98 metrics with explicit region-id mapping."""
    pred = prediction.float()
    true = target.float()
    label = labels.long()
    valid = brain_mask.float() > 0.5
    all_pred_means = []
    all_true_means = []
    all_pred_stds = []
    all_true_stds = []
    all_mae = []
    all_rmse = []
    edge_errors = []
    hemi_errors = []
    valid_region_counts = []

    for batch_index in range(pred.shape[0]):
        region_stats = {}
        for region_id in range(1, graph.num_regions + 1):
            region = (label[batch_index, 0] == region_id) & valid[batch_index, 0]
            if not region.any():
                continue
            p = pred[batch_index, 0][region]
            t = true[batch_index, 0][region]
            region_stats[region_id] = (p.mean(), t.mean(), p.std(unbiased=False), t.std(unbiased=False))
            all_mae.append((p - t).abs().mean())
            all_rmse.append((p - t).square().mean().sqrt())

        valid_region_counts.append(float(len(region_stats)))
        for region_id, (p_mean, t_mean, p_std, t_std) in region_stats.items():
            all_pred_means.append(p_mean)
            all_true_means.append(t_mean)
            all_pred_stds.append(p_std)
            all_true_stds.append(t_std)

        for left, right in graph.edges.tolist():
            if left not in region_stats or right not in region_stats:
                continue
            p_left, t_left = region_stats[left][0], region_stats[left][1]
            p_right, t_right = region_stats[right][0], region_stats[right][1]
            pred_rel = (p_left - p_right) / (p_left.abs() + p_right.abs() + 1e-6)
            true_rel = (t_left - t_right) / (t_left.abs() + t_right.abs() + 1e-6)
            edge_errors.append((pred_rel - true_rel).abs())

        for left, right in graph.homologous_edges.tolist():
            if left not in region_stats or right not in region_stats:
                continue
            p_left, t_left = region_stats[left][0], region_stats[left][1]
            p_right, t_right = region_stats[right][0], region_stats[right][1]
            pred_rel = (p_left - p_right) / (p_left.abs() + p_right.abs() + 1e-6)
            true_rel = (t_left - t_right) / (t_left.abs() + t_right.abs() + 1e-6)
            hemi_errors.append((pred_rel - true_rel).abs())

    if not all_mae:
        return {
            "region_macro_mae": float("nan"),
            "region_macro_rmse": float("nan"),
            "region_std_mae": float("nan"),
            "region_mean_pearson": float("nan"),
            "region_mean_spearman": float("nan"),
            "region_edge_relation_error": float("nan"),
            "region_hemisphere_error": float("nan"),
            "num_valid_regions": 0.0,
        }

    pred_means = torch.stack(all_pred_means)
    true_means = torch.stack(all_true_means)
    pred_stds = torch.stack(all_pred_stds)
    true_stds = torch.stack(all_true_stds)
    return {
        "region_macro_mae": float(torch.stack(all_mae).mean().item()),
        "region_macro_rmse": float(torch.stack(all_rmse).mean().item()),
        "region_std_mae": float((pred_stds - true_stds).abs().mean().item()),
        "region_mean_pearson": _corr(pred_means, true_means),
        "region_mean_spearman": _corr(pred_means, true_means, spearman=True),
        "region_edge_relation_error": float(torch.stack(edge_errors).mean().item()) if edge_errors else float("nan"),
        "region_hemisphere_error": float(torch.stack(hemi_errors).mean().item()) if hemi_errors else float("nan"),
        "num_valid_regions": sum(valid_region_counts) / len(valid_region_counts),
    }


def patch_boundary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> dict[str, float]:
    """Measure value and gradient errors in a two-voxel slab around each seam."""
    pred = prediction.float()
    true = target.float()
    value_errors = []
    gradient_errors = []
    for dimension, width in enumerate(patch_size):
        axis = 2 + dimension
        for boundary in range(int(width), pred.shape[axis], int(width)):
            for offset in (0, 1):
                left_index = boundary - 1 - offset
                right_index = boundary + offset
                if left_index < 0 or right_index >= pred.shape[axis]:
                    continue
                left = [slice(None)] * pred.ndim
                right = [slice(None)] * pred.ndim
                left[axis] = left_index
                right[axis] = right_index
                value_errors.append((pred[tuple(left)] - true[tuple(left)]).abs())
                value_errors.append((pred[tuple(right)] - true[tuple(right)]).abs())
                if offset == 0:
                    pred_jump = pred[tuple(right)] - pred[tuple(left)]
                    true_jump = true[tuple(right)] - true[tuple(left)]
                    gradient_errors.append((pred_jump - true_jump).abs())
    if not value_errors:
        return {"patch_boundary_mae": float("nan"), "patch_boundary_gradient_error": float("nan")}
    return {
        "patch_boundary_mae": float(torch.cat([item.reshape(-1) for item in value_errors]).mean().item()),
        "patch_boundary_gradient_error": float(torch.cat([item.reshape(-1) for item in gradient_errors]).mean().item())
        if gradient_errors else float("nan"),
    }
