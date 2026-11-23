"""Parc98 ROI mean-uptake extraction from the *original* PET target.

The region target must come from the raw GT PET + the Parc98 atlas, never from
the coarse prediction.  Empty ROIs (voxel count == 0) produce a mean of 0.0 and
a count of 0; callers should mask them out of the loss / Pearson correlation.
"""

from __future__ import annotations

import torch


def extract_region_uptake(
    pet: torch.Tensor,
    labels: torch.Tensor,
    num_regions: int = 98,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (means[B,98], counts[B,98]) of mean PET uptake per Parc98 ROI.

    labels are the integer Parc98 labelmap [B,1,D,H,W] (or [1,1,D,H,W] shared).
    Region id r (1..98) maps to column r-1.  Empty ROIs -> mean 0.0, count 0.
    """
    if pet.ndim != 5:
        raise ValueError(f"Expected PET [B,1,D,H,W], got {tuple(pet.shape)}")
    labels = labels.long()
    if labels.ndim == 5:
        pass
    elif labels.ndim == 4:
        # [1,D,H,W] shared labelmap (as emitted by Exp027Dataset)
        labels = labels[:, None]
    elif labels.ndim == 3:
        labels = labels[None, None]
    else:
        raise ValueError(
            f"Expected Parc98 labels [B,1,D,H,W], got {tuple(labels.shape)}"
        )
    if labels.shape[0] == 1 and pet.shape[0] > 1:
        labels = labels.expand(pet.shape[0], -1, -1, -1, -1)
    batch = pet.shape[0]
    means = []
    counts = []
    for region_id in range(1, int(num_regions) + 1):
        region = (labels == region_id).float()
        count = region.flatten(2).sum(dim=2)  # [B,1]
        region_sum = (region * pet).flatten(2).sum(dim=2)
        mean = region_sum / count.clamp_min(1e-6)
        means.append(mean[:, 0])
        counts.append(count[:, 0])
    return torch.stack(means, dim=1), torch.stack(counts, dim=1)


__all__ = ["extract_region_uptake"]
