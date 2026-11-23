from __future__ import annotations

from typing import Optional

import torch

from losses.diffusion_losses import ensure_mask
from metrics.image_metrics import ImageMetricCollection


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weight = ensure_mask(mask, a)
    return (torch.abs(a - b) * weight).sum() / weight.sum().clamp_min(float(eps))


def compute_masked_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    data_range: float = 2.0,
) -> dict[str, float]:
    metrics = ImageMetricCollection(data_range=data_range)
    if mask is None:
        return metrics(prediction, target, mask=None)
    weight = ensure_mask(mask, prediction)
    base = metrics(prediction, target, mask=weight)
    denom = weight.sum().clamp_min(1e-8)
    mse = ((((prediction.float() - target.float()) ** 2) * weight).sum() / denom).item()
    if mse <= 0:
        base["psnr"] = float("inf")
    else:
        base["psnr"] = float(10.0 * torch.log10(torch.tensor(float(data_range) ** 2) / torch.tensor(mse)).item())
    ssim_map = metrics.ssim(prediction.float(), target.float(), return_map=True)
    base["ssim"] = float(((ssim_map * weight).sum() / denom).item())
    return base
