from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Parc98Graph:
    num_regions: int
    edges: torch.Tensor
    homologous_edges: torch.Tensor
    region_weights: torch.Tensor
    included_regions: torch.Tensor


def _resize_labels(path: Path, shape: tuple[int, int, int]) -> torch.Tensor:
    values = nib.load(str(path)).get_fdata(dtype="float32")
    labels = torch.from_numpy(values).long()[None, None]
    if tuple(labels.shape[-3:]) != shape:
        labels = F.interpolate(labels.float(), size=shape, mode="nearest").long()
    return labels[0, 0]


def build_parc98_graph(
    labelmap_path: str | Path,
    image_size: tuple[int, int, int],
    *,
    num_regions: int = 98,
    excluded_regions: tuple[int, ...] = (2, 3, 10, 11, 15, 19, 20),
    auxiliary_regions: tuple[int, ...] = (1, 4, 12, 18, 21),
) -> Parc98Graph:
    labels = _resize_labels(Path(labelmap_path), image_size)
    pairs: set[tuple[int, int]] = set()
    for dimension in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[dimension] = slice(None, -1)
        second[dimension] = slice(1, None)
        left = labels[tuple(first)].reshape(-1)
        right = labels[tuple(second)].reshape(-1)
        different = (left != right) & (left > 0) & (right > 0)
        for a, b in zip(left[different].tolist(), right[different].tolist()):
            pair = (min(int(a), int(b)), max(int(a), int(b)))
            if pair[0] != pair[1] and pair[1] <= num_regions:
                pairs.add(pair)

    for left_id in range(31, 65):
        pairs.add((left_id, left_id + 34))

    edges = torch.tensor(sorted(pairs), dtype=torch.long)
    if edges.numel() == 0:
        edges = torch.empty((0, 2), dtype=torch.long)
    included = [
        region_id
        for region_id in range(1, num_regions + 1)
        if region_id not in set(excluded_regions)
    ]
    weights = torch.zeros(num_regions + 1, dtype=torch.float32)
    for region_id in included:
        weights[region_id] = 0.25 if region_id in set(auxiliary_regions) else 1.0
    included_tensor = torch.tensor(included, dtype=torch.long)
    weights[included_tensor] = weights[included_tensor] / weights[included_tensor].mean().clamp_min(1e-6)
    homologous = torch.tensor(
        [(left_id, left_id + 34) for left_id in range(31, 65)], dtype=torch.long
    )
    return Parc98Graph(
        num_regions=num_regions,
        edges=edges,
        homologous_edges=homologous,
        region_weights=weights,
        included_regions=included_tensor,
    )
