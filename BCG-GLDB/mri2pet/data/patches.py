from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


Coord3D = tuple[int, int, int]


def validate_patch_geometry(
    image_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(image_size) != 3 or len(patch_size) != 3:
        raise ValueError("image_size and patch_size must contain three dimensions")
    if any(int(v) <= 0 for v in (*image_size, *patch_size)):
        raise ValueError("image_size and patch_size must be positive")
    if any(int(image_size[i]) % int(patch_size[i]) != 0 for i in range(3)):
        raise ValueError(f"patch_size={patch_size} must divide image_size={image_size}")
    return tuple(int(image_size[i]) // int(patch_size[i]) for i in range(3))


def build_patch_grid(
    image_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> list[Coord3D]:
    validate_patch_geometry(image_size, patch_size)
    return [
        (z, y, x)
        for z in range(0, int(image_size[0]), int(patch_size[0]))
        for y in range(0, int(image_size[1]), int(patch_size[1]))
        for x in range(0, int(image_size[2]), int(patch_size[2]))
    ]


def extract_patches(
    volume: torch.Tensor,
    coords: Iterable[Coord3D],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    if volume.ndim != 5:
        raise ValueError(f"Expected volume [B,C,D,H,W], got {tuple(volume.shape)}")
    pd, ph, pw = [int(v) for v in patch_size]
    patches = [volume[..., z : z + pd, y : y + ph, x : x + pw] for z, y, x in coords]
    if not patches:
        raise ValueError("Patch coordinate list is empty")
    if any(tuple(p.shape[-3:]) != (pd, ph, pw) for p in patches):
        raise ValueError("A patch extends outside the input volume")
    return torch.stack(patches, dim=1)


def stitch_patches(
    patches: torch.Tensor,
    coords: Iterable[Coord3D],
    output_size: tuple[int, int, int],
) -> torch.Tensor:
    if patches.ndim != 6:
        raise ValueError(f"Expected patches [B,P,C,D,H,W], got {tuple(patches.shape)}")
    coords = list(coords)
    if patches.shape[1] != len(coords):
        raise ValueError("Patch count and coordinate count do not match")
    output = patches.new_zeros((patches.shape[0], patches.shape[2], *output_size))
    weights = patches.new_zeros((patches.shape[0], 1, *output_size))
    for index, (z, y, x) in enumerate(coords):
        patch = patches[:, index]
        d, h, w = patch.shape[-3:]
        output[..., z : z + d, y : y + h, x : x + w] += patch
        weights[..., z : z + d, y : y + h, x : x + w] += 1
    if torch.any(weights == 0):
        raise ValueError("Patch grid does not cover the complete output volume")
    return output / weights


def crop_feature_grid(
    feature: torch.Tensor,
    grid_coords: torch.Tensor,
    grid_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
) -> torch.Tensor:
    """Crop grid cells from a full feature map and resize every cell to one shape."""
    if feature.ndim != 5 or grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError("feature must be [B,C,D,H,W] and grid_coords must be [P,3]")
    spatial = feature.shape[-3:]
    cells = []
    for coord in grid_coords.long().tolist():
        starts = [int(coord[i] * spatial[i] // grid_size[i]) for i in range(3)]
        ends = [int((coord[i] + 1) * spatial[i] // grid_size[i]) for i in range(3)]
        cell = feature[
            ...,
            starts[0] : max(ends[0], starts[0] + 1),
            starts[1] : max(ends[1], starts[1] + 1),
            starts[2] : max(ends[2], starts[2] + 1),
        ]
        if tuple(cell.shape[-3:]) != tuple(output_size):
            cell = F.interpolate(cell, size=output_size, mode="trilinear", align_corners=False)
        cells.append(cell)
    return torch.stack(cells, dim=1)


def grid_indices(grid_size: tuple[int, int, int]) -> list[Coord3D]:
    return [
        (z, y, x)
        for z in range(int(grid_size[0]))
        for y in range(int(grid_size[1]))
        for x in range(int(grid_size[2]))
    ]
