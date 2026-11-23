from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from exp027.utils.parc98_graph import Parc98Graph


def _region_stats(
    value: torch.Tensor,
    labels: torch.Tensor,
    num_regions: int,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-region sum, squared sum and count without allocating one-hot maps."""
    if value.ndim != 5 or labels.ndim != 5:
        raise ValueError("value and labels must be [B,C,D,H,W]")
    if labels.shape[1] != 1 or value.shape[0] != labels.shape[0]:
        raise ValueError("labels must have one channel and match value batch")
    batch, channels = value.shape[:2]
    sums = value.new_zeros((batch, num_regions + 1, channels))
    squared = value.new_zeros((batch, num_regions + 1, channels))
    counts = value.new_zeros((batch, num_regions + 1))
    for batch_index in range(batch):
        region_ids = labels[batch_index, 0].reshape(-1).long()
        values = value[batch_index].permute(1, 2, 3, 0).reshape(-1, channels)
        valid = (region_ids > 0) & (region_ids <= num_regions)
        if mask is not None:
            valid = valid & (mask[batch_index, 0].reshape(-1) > 0.5)
        region_ids = region_ids[valid]
        values = values[valid]
        if region_ids.numel() == 0:
            continue
        counts[batch_index].scatter_add_(0, region_ids, values.new_ones(region_ids.shape))
        sums[batch_index].scatter_add_(
            0, region_ids[:, None].expand(-1, channels), values
        )
        squared[batch_index].scatter_add_(
            0, region_ids[:, None].expand(-1, channels), values.square()
        )
    return sums, squared, counts


def _region_means(
    value: torch.Tensor,
    labels: torch.Tensor,
    num_regions: int,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sums, squared, counts = _region_stats(value, labels, num_regions, mask)
    denominator = counts.clamp_min(1.0).unsqueeze(-1)
    means = sums / denominator
    variance = (squared / denominator - means.square()).clamp_min(0.0)
    return means, variance.sqrt(), counts


def regional_metabolic_graph_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    graph: Parc98Graph,
    brain_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Node, adjacent-region and left/right metabolic consistency."""
    pred_mean, pred_std, pred_count = _region_means(
        prediction, labels, graph.num_regions, brain_mask
    )
    target_mean, target_std, target_count = _region_means(
        target, labels, graph.num_regions, brain_mask
    )
    active = graph.included_regions.to(device=prediction.device)
    weights = graph.region_weights.to(device=prediction.device)[active]
    valid = (pred_count[:, active] > 0) & (target_count[:, active] > 0)
    valid = valid.float().unsqueeze(-1)
    denom = (weights[None, :, None] * valid).sum().clamp_min(1.0)
    node = (
        weights[None, :, None]
        * valid
        * ((pred_mean[:, active] - target_mean[:, active]).abs()
           + 0.25 * (pred_std[:, active] - target_std[:, active]).abs())
    ).sum() / denom

    edge_terms = []
    edge_count = []
    eps = prediction.new_tensor(1e-6)
    for edge in graph.edges.to(device=prediction.device):
        left, right = int(edge[0]), int(edge[1])
        if left not in set(active.tolist()) or right not in set(active.tolist()):
            continue
        pair_valid = (pred_count[:, left] > 0) & (pred_count[:, right] > 0)
        target_pair_valid = (target_count[:, left] > 0) & (target_count[:, right] > 0)
        pair_valid = pair_valid & target_pair_valid
        if not pair_valid.any():
            continue
        pred_rel = (pred_mean[:, left, 0] - pred_mean[:, right, 0]) / (
            pred_mean[:, left, 0].abs() + pred_mean[:, right, 0].abs() + eps
        )
        target_rel = (target_mean[:, left, 0] - target_mean[:, right, 0]) / (
            target_mean[:, left, 0].abs() + target_mean[:, right, 0].abs() + eps
        )
        edge_terms.append((pred_rel - target_rel).abs()[pair_valid].mean())
        edge_count.append(torch.sqrt(
            graph.region_weights[left] * graph.region_weights[right]
        ).to(prediction.device))
    edge_loss = (
        (torch.stack(edge_terms) * torch.stack(edge_count)).sum()
        / torch.stack(edge_count).sum().clamp_min(1e-6)
        if edge_terms
        else prediction.new_zeros(())
    )

    hemisphere_terms = []
    for left, right in graph.homologous_edges.to(device=prediction.device).tolist():
        pair_valid = (pred_count[:, left] > 0) & (pred_count[:, right] > 0)
        pair_valid = pair_valid & (target_count[:, left] > 0) & (target_count[:, right] > 0)
        if not pair_valid.any():
            continue
        pred_asym = (pred_mean[:, left, 0] - pred_mean[:, right, 0]) / (
            pred_mean[:, left, 0].abs() + pred_mean[:, right, 0].abs() + eps
        )
        target_asym = (target_mean[:, left, 0] - target_mean[:, right, 0]) / (
            target_mean[:, left, 0].abs() + target_mean[:, right, 0].abs() + eps
        )
        hemisphere_terms.append((pred_asym - target_asym).abs()[pair_valid].mean())
    hemisphere = (
        torch.stack(hemisphere_terms).mean()
        if hemisphere_terms
        else prediction.new_zeros(())
    )
    total = node + 0.5 * edge_loss + 0.25 * hemisphere
    return total, {
        "node": node.detach(),
        "edge": edge_loss.detach(),
        "hemisphere": hemisphere.detach(),
        "rmgc": total.detach(),
    }


def patch_to_region_alignment_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    latent_labels: torch.Tensor,
    graph: Parc98Graph,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_mean, _, pred_count = _region_means(
        predicted_latent, latent_labels, graph.num_regions
    )
    target_mean, _, target_count = _region_means(
        target_latent, latent_labels, graph.num_regions
    )
    active = graph.included_regions.to(device=predicted_latent.device)
    weights = graph.region_weights.to(device=predicted_latent.device)[active]
    valid = ((pred_count[:, active] > 0) & (target_count[:, active] > 0)).float()
    valid = valid.unsqueeze(-1)
    denom = (valid * weights[None, :, None]).sum().clamp_min(1.0)
    loss = (
        valid
        * weights[None, :, None]
        * (pred_mean[:, active] - target_mean[:, active]).abs()
    ).sum() / denom

    relation_terms = []
    active_set = set(active.tolist())
    for left, right in graph.edges.to(device=predicted_latent.device).tolist():
        if left not in active_set or right not in active_set:
            continue
        valid_pair = (pred_count[:, left] > 0) & (pred_count[:, right] > 0)
        valid_pair = valid_pair & (target_count[:, left] > 0) & (target_count[:, right] > 0)
        if valid_pair.any():
            relation_terms.append(
                (
                    (pred_mean[:, left] - pred_mean[:, right])
                    - (target_mean[:, left] - target_mean[:, right])
                ).abs()[valid_pair].mean()
            )
    relation = (
        torch.stack(relation_terms).mean()
        if relation_terms
        else predicted_latent.new_zeros(())
    )
    total = loss + 0.5 * relation
    return total, {"pra": loss.detach(), "lgr": relation.detach(), "latent_region": total.detach()}


def patch_boundary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Match first-order jumps across the three patch boundary directions."""
    terms = []
    for dimension, patch_width in enumerate(patch_size):
        for boundary in range(int(patch_width), prediction.shape[2 + dimension], int(patch_width)):
            for offset in (0, 1):
                left_index = boundary - 1 - offset
                right_index = boundary + offset
                if left_index < 0 or right_index >= prediction.shape[2 + dimension]:
                    continue
                left = [slice(None)] * prediction.ndim
                right = [slice(None)] * prediction.ndim
                left[2 + dimension] = left_index
                right[2 + dimension] = right_index
                pred_jump = prediction[tuple(right)] - prediction[tuple(left)]
                target_jump = target[tuple(right)] - target[tuple(left)]
                terms.append(F.l1_loss(pred_jump, target_jump))
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


# Keep the historical public names numerically aligned with Exp027's soft-occupancy
# implementation.  The trainer imports the explicit names directly, but external
# evaluation scripts may still import these compatibility names.
from exp027.losses.soft_region import (
    regional_metabolic_graph_loss_soft as regional_metabolic_graph_loss,
    soft_patch_to_region_alignment_loss as patch_to_region_alignment_loss,
)
