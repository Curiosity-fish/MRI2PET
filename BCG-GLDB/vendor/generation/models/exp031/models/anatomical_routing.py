"""Patch-to-Region anatomical routing.

For the 64 image patches (4x4x4 grid, patch = 40x48x40) and the 98 Parc98 ROIs,
the routing matrix A is

    A[i,j] = voxel_overlap(Patch_i, ROI_j) / voxel_count(Patch_i)

optionally row-normalised so every patch's routing over the ROIs sums to 1
(rows with no overlap at all are left as zero rows).  A is precomputed once
from the fixed Parc98 labelmap and cached to disk; it is never recomputed per
iteration.  In attention we use a *soft* bias  log(A + eps)  (masked to -inf
where A == 0) instead of hard one-hot routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from exp026.utils.patch_codec import build_patch_grid, validate_patch_geometry


def build_routing_matrix(
    labels: torch.Tensor,
    image_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    num_regions: int = 98,
    normalize_rows: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute [num_patches, num_regions] fractional-overlap routing matrix.

    labels: [1,1,D,H,W] (or [D,H,W]) integer Parc98 labelmap at image_size.
    """
    if labels.ndim == 5:
        labels = labels[0, 0]
    elif labels.ndim == 4:
        labels = labels[0]
    if tuple(labels.shape) != tuple(int(v) for v in image_size):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} != image_size {tuple(image_size)}"
        )
    grid_size = validate_patch_geometry(image_size, patch_size)
    coords = build_patch_grid(image_size, patch_size)
    num_patches = len(coords)
    pd, ph, pw = [int(v) for v in patch_size]
    labels = labels.to(device=device)
    routing = torch.zeros((num_patches, int(num_regions)), dtype=torch.float32, device=device)
    for patch_index, (z, y, x) in enumerate(coords):
        patch_labels = labels[z : z + pd, y : y + ph, x : x + pw].reshape(-1)
        counts = torch.bincount(
            patch_labels, minlength=int(num_regions) + 1
        ).float()
        # region j stored in column j-1 (region 0 = background is ignored)
        routing[patch_index] = counts[1 : int(num_regions) + 1]
    routing = routing / float(pd * ph * pw)
    if bool(normalize_rows):
        row_sum = routing.sum(dim=1, keepdim=True)
        routing = torch.where(
            row_sum > 0,
            routing / row_sum.clamp_min(1e-8),
            torch.zeros_like(routing),
        )
    return routing.contiguous()


def build_routing_matrix_from_path(
    labelmap_path: str | Path,
    image_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    num_regions: int = 98,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Load a Parc98 labelmap nii.gz and compute the routing matrix on CPU."""
    import nibabel as nib

    values = nib.load(str(Path(labelmap_path))).get_fdata(dtype="float32")
    labels = torch.from_numpy(values).long()
    if tuple(labels.shape) != tuple(int(v) for v in image_size):
        import torch.nn.functional as F

        labels = F.interpolate(
            labels.float()[None, None], size=tuple(int(v) for v in image_size),
            mode="nearest",
        ).long()[0, 0]
    return build_routing_matrix(
        labels, image_size, patch_size, num_regions=num_regions,
        normalize_rows=normalize_rows,
    )


def save_routing_matrix(matrix: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(matrix.cpu(), path)


def load_routing_matrix(path: str | Path) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu")


def get_or_build_routing_matrix(
    labelmap_path: str | Path,
    image_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    cache_path: Optional[str | Path] = None,
    num_regions: int = 98,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Return the cached routing matrix, building + saving it on first use."""
    if cache_path is not None and Path(cache_path).exists():
        return load_routing_matrix(cache_path)
    matrix = build_routing_matrix_from_path(
        labelmap_path, image_size, patch_size,
        num_regions=num_regions, normalize_rows=normalize_rows,
    )
    if cache_path is not None:
        save_routing_matrix(matrix, cache_path)
    return matrix


def routing_attention_bias(
    routing: torch.Tensor,
    patch_indices: torch.Tensor,
    eps: float = 1e-6,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """Soft attention bias log(A + eps) masked to -inf where A == 0.

    routing:        [num_patches, num_regions]
    patch_indices:  [B, P] absolute patch indices (or [P])
    Returns [B, P, num_regions] additive attention logit bias.
    """
    selected = routing[patch_indices.long()]  # [B,P,R]
    bias = torch.log(selected + eps)
    bias = torch.where(selected > 0, bias, torch.full_like(bias, float(mask_value)))
    return bias
