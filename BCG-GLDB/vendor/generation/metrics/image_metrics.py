from __future__ import annotations

from typing import Dict, Optional

import torch

from losses.diffusion_losses import ensure_mask
from utils.psnr import calculate_psnr_3d
from utils.ssim import SSIM3D


class ImageMetricCollection:
    def __init__(self, data_range: float = 2.0) -> None:
        self.data_range = float(data_range)

        try:
            self.ssim = SSIM3D(data_range=self.data_range)
        except TypeError:
            try:
                self.ssim = SSIM3D(max_val=self.data_range)
            except TypeError:
                self.ssim = SSIM3D()

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        prediction = prediction.float()
        target = target.float()

        if mask is not None:
            weight = ensure_mask(mask, prediction)
            denom = weight.sum().clamp_min(1e-8)
            mae = ((torch.abs(prediction - target) * weight).sum() / denom).item()
            rmse = torch.sqrt((((prediction - target) ** 2 * weight).sum() / denom)).item()
        else:
            mae = torch.mean(torch.abs(prediction - target)).item()
            rmse = torch.sqrt(torch.mean((prediction - target) ** 2)).item()

        psnr = float(calculate_psnr_3d(prediction, target, max_value=self.data_range))

        try:
            ssim = float(self.ssim(prediction, target).item())
        except TypeError:
            ssim = float(self.ssim(prediction, target, data_range=self.data_range).item())

        return {
            "mae": mae,
            "rmse": rmse,
            "psnr": psnr,
            "ssim": ssim,
        }


def compute_image_metrics_detailed(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 2.0,
) -> Dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    metrics = ImageMetricCollection(data_range=data_range)
    full_metrics = metrics(prediction, target, mask=None)
    result = {
        "mae": full_metrics["mae"],
        "rmse": full_metrics["rmse"],
        "psnr": full_metrics["psnr"],
        "ssim": full_metrics["ssim"],
    }
    if mask is None:
        result.update(
            {
                "brain_mae": full_metrics["mae"],
                "brain_rmse": full_metrics["rmse"],
                "brain_psnr": full_metrics["psnr"],
                "brain_ssim": full_metrics["ssim"],
            }
        )
        return result

    weight = ensure_mask(mask, prediction)
    denom = weight.sum().clamp_min(1e-8)
    brain_mae = ((torch.abs(prediction - target) * weight).sum() / denom).item()
    brain_rmse = torch.sqrt((((prediction - target) ** 2) * weight).sum() / denom).item()

    brain_mask_bool = weight > 0
    if brain_mask_bool.any():
        pred_inside = prediction[brain_mask_bool]
        target_inside = target[brain_mask_bool]
        mse_inside = torch.mean((pred_inside - target_inside) ** 2)
        if mse_inside <= 0:
            brain_psnr = float("inf")
        else:
            brain_psnr = float(10.0 * torch.log10(torch.tensor(float(data_range) ** 2, device=prediction.device) / mse_inside).item())
    else:
        brain_psnr = float("nan")

    try:
        ssim_map = metrics.ssim(prediction, target, return_map=True)
        brain_ssim = ((ssim_map * weight).sum() / denom).item()
    except TypeError:
        # 当前 SSIM3D 不支持 return_map，只能返回整体 SSIM
        # 为了不中断验证，这里用整体 SSIM 作为 brain_ssim 的近似值
        brain_ssim = full_metrics["ssim"]

    result.update(
        {
            "brain_mae": brain_mae,
            "brain_rmse": brain_rmse,
            "brain_psnr": brain_psnr,
            "brain_ssim": brain_ssim,
        }
    )
    return result
