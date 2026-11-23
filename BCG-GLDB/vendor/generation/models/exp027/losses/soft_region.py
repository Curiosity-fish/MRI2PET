"""Soft Parc98 region losses and evaluation helpers.

Parc98 is converted to fractional occupancy at latent resolution by average
pooling one-hot masks. This preserves small regions and mixed patch boundaries.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from exp027.utils.parc98_graph import Parc98Graph


def _occupancy(labels: torch.Tensor, target_size: tuple[int, int, int], num_regions: int) -> torch.Tensor:
    """Build fractional occupancy without materializing a full-resolution one-hot map."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("Parc98 labels must be [B,1,D,H,W]")
    labels = labels.long().clamp(0, num_regions)
    channels = []
    for region_id in range(num_regions + 1):
        region = (labels == region_id).float()
        if tuple(labels.shape[-3:]) != tuple(target_size):
            region = F.interpolate(region, size=target_size, mode="area")
        channels.append(region)
    return torch.cat(channels, dim=1)


def _soft_region_means(
    value: torch.Tensor,
    occupancy: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if value.ndim != 5 or occupancy.ndim != 5:
        raise ValueError("value and occupancy must be 5D tensors")
    weights = occupancy
    if mask is not None:
        weights = weights * mask.float()
    counts = weights.flatten(2).sum(dim=2)
    sums = torch.einsum("brdhw,bcdhw->brc", weights, value)
    means = sums / counts.unsqueeze(-1).clamp_min(1e-6)
    squared = torch.einsum("brdhw,bcdhw->brc", weights, value.square())
    std = (squared / counts.unsqueeze(-1).clamp_min(1e-6) - means.square()).clamp_min(0).sqrt()
    return means, std, counts


def _soft_region_means_from_labels(
    value: torch.Tensor,
    labels: torch.Tensor,
    num_regions: int,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute soft region statistics with bounded memory for full-resolution labels."""
    target_size = tuple(value.shape[-3:])
    means = []
    stds = []
    counts = []
    for region_id in range(num_regions + 1):
        region = (labels.long() == region_id).float()
        if tuple(labels.shape[-3:]) != target_size:
            region = F.interpolate(region, size=target_size, mode="area")
        if mask is not None:
            region = region * mask.float()
        count = region.flatten(2).sum(dim=2)
        denominator = count.clamp_min(1e-6).unsqueeze(-1)
        region_sum = (region * value).flatten(2).sum(dim=2)
        region_squared = (region * value.square()).flatten(2).sum(dim=2)
        mean = region_sum / denominator
        std = (region_squared / denominator - mean.square()).clamp_min(0).sqrt()
        means.append(mean)
        stds.append(std)
        counts.append(count)
    return torch.stack(means, dim=1), torch.stack(stds, dim=1), torch.stack(counts, dim=1)

def soft_patch_to_region_alignment_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    parc98_labels: torch.Tensor,
    graph: Parc98Graph,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_mean, _, pred_count = _soft_region_means_from_labels(
        predicted_latent, parc98_labels, graph.num_regions
    )
    target_mean, _, target_count = _soft_region_means_from_labels(
        target_latent, parc98_labels, graph.num_regions
    )
    active = graph.included_regions.to(predicted_latent.device)
    weights = graph.region_weights.to(predicted_latent.device)[active]
    valid = ((pred_count[:, active] > 1e-6) & (target_count[:, active] > 1e-6)).float().unsqueeze(-1)
    denom = (valid * weights[None, :, None]).sum().clamp_min(1e-6)
    pra = (valid * weights[None, :, None] * (pred_mean[:, active] - target_mean[:, active]).abs()).sum() / denom

    edge_terms = []
    edge_weights = []
    active_set = set(active.tolist())
    for left, right in graph.edges.to(predicted_latent.device).tolist():
        if left not in active_set or right not in active_set:
            continue
        valid_pair = (pred_count[:, left] > 1e-6) & (pred_count[:, right] > 1e-6)
        valid_pair = valid_pair & (target_count[:, left] > 1e-6) & (target_count[:, right] > 1e-6)
        if valid_pair.any():
            relation = ((pred_mean[:, left] - pred_mean[:, right]) - (target_mean[:, left] - target_mean[:, right])).abs()
            edge_weight = torch.sqrt(graph.region_weights[left] * graph.region_weights[right]).to(predicted_latent.device)
            edge_terms.append(edge_weight * relation[valid_pair].mean())
            edge_weights.append(edge_weight)
    if edge_terms:
        lgr = torch.stack(edge_terms).sum() / torch.stack(edge_weights).sum().clamp_min(1e-6)
    else:
        lgr = predicted_latent.new_zeros(())
    total = pra + 0.5 * lgr
    return total, {
        "pra": pra.detach(),
        "lgr": lgr.detach(),
        "latent_region": total.detach(),
        "pra_raw": pra,
        "lgr_raw": lgr,
    }


def regional_metabolic_graph_loss_soft(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    graph: Parc98Graph,
    brain_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_mean, pred_std, pred_count = _soft_region_means_from_labels(
        prediction, labels, graph.num_regions, brain_mask
    )
    target_mean, target_std, target_count = _soft_region_means_from_labels(
        target, labels, graph.num_regions, brain_mask
    )
    active = graph.included_regions.to(prediction.device)
    weights = graph.region_weights.to(prediction.device)[active]
    valid = ((pred_count[:, active] > 1e-6) & (target_count[:, active] > 1e-6)).float().unsqueeze(-1)
    denom = (weights[None, :, None] * valid).sum().clamp_min(1e-6)
    node = (weights[None, :, None] * valid * ((pred_mean[:, active] - target_mean[:, active]).abs() + 0.25 * (pred_std[:, active] - target_std[:, active]).abs())).sum() / denom
    eps = prediction.new_tensor(1e-6)
    edge_terms = []
    edge_weights = []
    active_set = set(active.tolist())
    for left, right in graph.edges.to(prediction.device).tolist():
        if left not in active_set or right not in active_set:
            continue
        valid_pair = (pred_count[:, left] > 1e-6) & (pred_count[:, right] > 1e-6)
        valid_pair = valid_pair & (target_count[:, left] > 1e-6) & (target_count[:, right] > 1e-6)
        if not valid_pair.any():
            continue
        pred_rel = (pred_mean[:, left, 0] - pred_mean[:, right, 0]) / (pred_mean[:, left, 0].abs() + pred_mean[:, right, 0].abs() + eps)
        target_rel = (target_mean[:, left, 0] - target_mean[:, right, 0]) / (target_mean[:, left, 0].abs() + target_mean[:, right, 0].abs() + eps)
        edge_terms.append((pred_rel - target_rel).abs()[valid_pair].mean())
        edge_weights.append(torch.sqrt(graph.region_weights[left] * graph.region_weights[right]).to(prediction.device))
    edge = (torch.stack(edge_terms) * torch.stack(edge_weights)).sum() / torch.stack(edge_weights).sum().clamp_min(1e-6) if edge_terms else prediction.new_zeros(())
    hemisphere_terms = []
    for left, right in graph.homologous_edges.to(prediction.device).tolist():
        valid_pair = (pred_count[:, left] > 1e-6) & (pred_count[:, right] > 1e-6) & (target_count[:, left] > 1e-6) & (target_count[:, right] > 1e-6)
        if valid_pair.any():
            pred_asym = (pred_mean[:, left, 0] - pred_mean[:, right, 0]) / (pred_mean[:, left, 0].abs() + pred_mean[:, right, 0].abs() + eps)
            target_asym = (target_mean[:, left, 0] - target_mean[:, right, 0]) / (target_mean[:, left, 0].abs() + target_mean[:, right, 0].abs() + eps)
            hemisphere_terms.append((pred_asym - target_asym).abs()[valid_pair].mean())
    hemisphere = torch.stack(hemisphere_terms).mean() if hemisphere_terms else prediction.new_zeros(())
    total = node + 0.5 * edge + 0.25 * hemisphere
    return total, {"node": node.detach(), "edge": edge.detach(), "hemisphere": hemisphere.detach(), "rmgc": total.detach()}
