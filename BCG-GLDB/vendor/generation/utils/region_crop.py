from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropSlices:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.z0, self.z1, self.y0, self.y1, self.x0, self.x1)


@dataclass(frozen=True)
class PadInfo:
    pad_d_before: int
    pad_d_after: int
    pad_h_before: int
    pad_h_after: int
    pad_w_before: int
    pad_w_after: int

    def as_fpad(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.pad_w_before,
            self.pad_w_after,
            self.pad_h_before,
            self.pad_h_after,
            self.pad_d_before,
            self.pad_d_after,
        )

    def total(self) -> tuple[int, int, int]:
        return (
            self.pad_d_before + self.pad_d_after,
            self.pad_h_before + self.pad_h_after,
            self.pad_w_before + self.pad_w_after,
        )


def _compute_axis_bounds(
    low: int,
    high: int,
    full_size: int,
    margin: int,
    min_size: int,
    pad_to_multiple: int,
) -> tuple[int, int]:
    start = max(0, int(low) - int(margin))
    end = min(full_size, int(high) + 1 + int(margin))
    size = end - start

    if size < int(min_size):
        extra = int(min_size) - size
        before = extra // 2
        after = extra - before
        start = max(0, start - before)
        end = min(full_size, end + after)
        size = end - start
        if size < int(min_size):
            missing = int(min_size) - size
            if start == 0:
                end = min(full_size, end + missing)
            else:
                start = max(0, start - missing)
        size = end - start

    remainder = size % int(pad_to_multiple)
    if remainder != 0:
        extra = int(pad_to_multiple) - remainder
        before = extra // 2
        after = extra - before
        start = max(0, start - before)
        end = min(full_size, end + after)
        size = end - start
        remainder = size % int(pad_to_multiple)
        if remainder != 0:
            if start == 0:
                end = min(full_size, end + (int(pad_to_multiple) - remainder))
            elif end == full_size:
                start = max(0, start - (int(pad_to_multiple) - remainder))
    return int(start), int(end)


def compute_crop_slices_from_mask(
    mask: torch.Tensor,
    margin: Iterable[int],
    min_size: Iterable[int],
    pad_to_multiple: int,
) -> CropSlices:
    if mask.ndim != 4 or mask.shape[0] != 1:
        raise ValueError(f"Expected mask shape [1, D, H, W], got {tuple(mask.shape)}")
    coords = torch.nonzero(mask[0] > 0, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("Region mask is empty.")

    margin = tuple(int(v) for v in margin)
    min_size = tuple(int(v) for v in min_size)
    d, h, w = mask.shape[-3:]
    mins = coords.min(dim=0).values.tolist()
    maxs = coords.max(dim=0).values.tolist()

    z0, z1 = _compute_axis_bounds(mins[0], maxs[0], d, margin[0], min_size[0], pad_to_multiple)
    y0, y1 = _compute_axis_bounds(mins[1], maxs[1], h, margin[1], min_size[1], pad_to_multiple)
    x0, x1 = _compute_axis_bounds(mins[2], maxs[2], w, margin[2], min_size[2], pad_to_multiple)
    return CropSlices(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)


def crop_volume(volume: torch.Tensor, crop_slices: CropSlices) -> torch.Tensor:
    if volume.ndim != 4:
        raise ValueError(f"Expected volume shape [C, D, H, W], got {tuple(volume.shape)}")
    return volume[:, crop_slices.z0:crop_slices.z1, crop_slices.y0:crop_slices.y1, crop_slices.x0:crop_slices.x1]


def compute_required_padding(
    shape: tuple[int, int, int],
    pad_to_multiple: int,
) -> PadInfo:
    pads = []
    for size in shape:
        remainder = int(size) % int(pad_to_multiple)
        extra = 0 if remainder == 0 else int(pad_to_multiple) - remainder
        before = extra // 2
        after = extra - before
        pads.append((before, after))
    return PadInfo(
        pad_d_before=pads[0][0],
        pad_d_after=pads[0][1],
        pad_h_before=pads[1][0],
        pad_h_after=pads[1][1],
        pad_w_before=pads[2][0],
        pad_w_after=pads[2][1],
    )


def apply_padding(volume: torch.Tensor, pad_info: PadInfo, value: float = 0.0) -> torch.Tensor:
    return F.pad(volume, pad=pad_info.as_fpad(), mode="constant", value=float(value))


def remove_padding(volume: torch.Tensor, pad_info: PadInfo) -> torch.Tensor:
    d0 = pad_info.pad_d_before
    d1 = volume.shape[-3] - pad_info.pad_d_after
    h0 = pad_info.pad_h_before
    h1 = volume.shape[-2] - pad_info.pad_h_after
    w0 = pad_info.pad_w_before
    w1 = volume.shape[-1] - pad_info.pad_w_after
    return volume[:, d0:d1, h0:h1, w0:w1]
