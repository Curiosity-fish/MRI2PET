from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.region_crop import CropSlices


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(float(sigma), 1e-6)
    radius = max(int(round(3.0 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def gaussian_smooth_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return mask.float()
    kernel = _gaussian_kernel1d(sigma, device=mask.device, dtype=mask.dtype)
    radius = (kernel.numel() - 1) // 2

    out = mask.float().unsqueeze(0)
    kd = kernel.view(1, 1, -1, 1, 1)
    kh = kernel.view(1, 1, 1, -1, 1)
    kw = kernel.view(1, 1, 1, 1, -1)
    out = F.conv3d(out, kd, padding=(radius, 0, 0))
    out = F.conv3d(out, kh, padding=(0, radius, 0))
    out = F.conv3d(out, kw, padding=(0, 0, radius))
    out = out.squeeze(0)
    max_value = out.max().clamp_min(1e-8)
    return out / max_value


@dataclass
class StitchAccumulator:
    full_shape: tuple[int, int, int]
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        self.pred_sum = torch.zeros((1,) + self.full_shape, device=self.device, dtype=self.dtype)
        self.weight_sum = torch.zeros((1,) + self.full_shape, device=self.device, dtype=self.dtype)

    def add_crop(
        self,
        prediction_crop: torch.Tensor,
        weight_crop: torch.Tensor,
        crop_slices: CropSlices,
    ) -> None:
        z0, z1, y0, y1, x0, x1 = crop_slices.as_tuple()
        self.pred_sum[:, z0:z1, y0:y1, x0:x1] += prediction_crop * weight_crop
        self.weight_sum[:, z0:z1, y0:y1, x0:x1] += weight_crop

    def finalize(self, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.pred_sum / self.weight_sum.clamp_min(float(eps))
        return prediction, self.weight_sum
