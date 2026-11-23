from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from utils.ssim import SSIM3D


class AFMRSSIMLoss(torch.nn.Module):
    """Anatomy-aware focal multi-scale residual SSIM loss."""

    def __init__(
        self,
        *,
        window_size: int = 11,
        data_range: float = 2.0,
        half_scale: int = 2,
        scale_weights: Sequence[float] = (0.7, 0.3),
        focal_gamma: float = 1.5,
        focal_strength: float = 2.0,
        focal_ssim_weight: float = 0.5,
        saliency_window: int = 5,
        variance_weight: float = 0.75,
        gradient_weight: float = 0.5,
        saliency_clip: float = 4.0,
        cross_scale_weight: float = 0.2,
        residual_ncc_weight: float = 0.35,
        residual_ncc_window: int = 7,
        residual_variance_floor: float = 0.25,
        global_ssim_weight: float = 0.5,
        brain_ssim_weight: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(scale_weights) != 2:
            raise ValueError("scale_weights must contain full and half weights")
        if saliency_window % 2 == 0 or residual_ncc_window % 2 == 0:
            raise ValueError("Local-statistic window sizes must be odd")
        self.full_ssim = SSIM3D(
            window_size=window_size, data_range=data_range
        )
        self.half_ssim = SSIM3D(
            window_size=window_size, data_range=data_range
        )
        self.half_scale = max(1, int(half_scale))
        weights = torch.tensor(scale_weights, dtype=torch.float32)
        self.register_buffer("scale_weights", weights / weights.sum())
        self.focal_gamma = float(focal_gamma)
        self.focal_strength = float(focal_strength)
        self.focal_ssim_weight = float(focal_ssim_weight)
        self.saliency_window = int(saliency_window)
        self.variance_weight = float(variance_weight)
        self.gradient_weight = float(gradient_weight)
        self.saliency_clip = float(saliency_clip)
        self.cross_scale_weight = float(cross_scale_weight)
        self.residual_ncc_weight = float(residual_ncc_weight)
        self.residual_ncc_window = int(residual_ncc_window)
        self.residual_variance_floor = float(residual_variance_floor)
        self.global_ssim_weight = float(global_ssim_weight)
        self.brain_ssim_weight = float(brain_ssim_weight)
        self.eps = float(eps)

    def _masked_mean(
        self, value: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        dimensions = tuple(range(1, value.ndim))
        numerator = (value * weight).sum(dim=dimensions)
        denominator = weight.sum(dim=dimensions).clamp_min(self.eps)
        return (numerator / denominator).mean()

    def _normalize_inside_mask(
        self, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        dimensions = tuple(range(1, value.ndim))
        mean = (value * mask).sum(dim=dimensions, keepdim=True)
        mean = mean / mask.sum(dim=dimensions, keepdim=True).clamp_min(
            self.eps
        )
        return (value / mean.clamp_min(self.eps)).clamp(
            min=0.0, max=self.saliency_clip
        )

    @staticmethod
    def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
        magnitude = torch.zeros_like(image)
        for dimension in (-3, -2, -1):
            difference = torch.diff(image, dim=dimension).abs()
            padding = [0, 0, 0, 0, 0, 0]
            axis = image.ndim + dimension
            padding[2 * (image.ndim - axis - 1) + 1] = 1
            magnitude = magnitude + F.pad(difference, padding)
        return magnitude / 3.0

    def _anatomy_weight(
        self, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        padding = self.saliency_window // 2
        mean = F.avg_pool3d(
            target,
            kernel_size=self.saliency_window,
            stride=1,
            padding=padding,
        )
        second_moment = F.avg_pool3d(
            target.square(),
            kernel_size=self.saliency_window,
            stride=1,
            padding=padding,
        )
        local_std = (second_moment - mean.square()).clamp_min(0.0).sqrt()
        gradient = self._gradient_magnitude(target)
        normalized_std = self._normalize_inside_mask(local_std, mask)
        normalized_gradient = self._normalize_inside_mask(gradient, mask)
        saliency = (
            1.0
            + self.variance_weight * normalized_std
            + self.gradient_weight * normalized_gradient
        )
        return (saliency * mask).detach()

    def _focal_ssim_term(
        self,
        error_map: torch.Tensor,
        mask: torch.Tensor,
        anatomy_weight: torch.Tensor,
    ) -> torch.Tensor:
        hard_weight = error_map.detach().clamp_min(0.0).add(
            self.eps
        ).pow(self.focal_gamma)
        weight = anatomy_weight * (
            1.0 + self.focal_strength * hard_weight
        )
        return self._masked_mean(error_map, weight * mask)

    def _local_residual_ncc(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        atlas: torch.Tensor,
        mask: torch.Tensor,
        anatomy_weight: torch.Tensor,
    ) -> torch.Tensor:
        prediction_residual = prediction - atlas
        target_residual = target - atlas
        window = self.residual_ncc_window
        padding = window // 2

        prediction_mean = F.avg_pool3d(
            prediction_residual, window, stride=1, padding=padding
        )
        target_mean = F.avg_pool3d(
            target_residual, window, stride=1, padding=padding
        )
        cross_moment = F.avg_pool3d(
            prediction_residual * target_residual,
            window,
            stride=1,
            padding=padding,
        )
        prediction_second_moment = F.avg_pool3d(
            prediction_residual.square(),
            window,
            stride=1,
            padding=padding,
        )
        target_second_moment = F.avg_pool3d(
            target_residual.square(),
            window,
            stride=1,
            padding=padding,
        )
        covariance = cross_moment - prediction_mean * target_mean
        prediction_variance = (
            prediction_second_moment - prediction_mean.square()
        ).clamp_min(0.0)
        target_variance = (
            target_second_moment - target_mean.square()
        ).clamp_min(0.0)
        ncc = covariance / (
            prediction_variance * target_variance + self.eps
        ).sqrt()
        ncc_error = 0.5 * (1.0 - ncc.clamp(-1.0, 1.0))

        residual_std = target_variance.detach().sqrt()
        residual_saliency = self._normalize_inside_mask(
            residual_std, mask
        ).detach()
        weight = anatomy_weight * (
            self.residual_variance_floor + residual_saliency
        )
        return self._masked_mean(ncc_error, weight * mask)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        atlas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        target = target.float()
        mask = mask.float()
        atlas = atlas.float()

        anatomy_weight = self._anatomy_weight(target, mask)
        full_map = self.full_ssim(prediction, target, return_map=True)
        full_error = 1.0 - full_map.clamp(-1.0, 1.0)
        full_focal = self._focal_ssim_term(
            full_error, mask, anatomy_weight
        )

        factor = self.half_scale
        if factor > 1:
            half_prediction = F.avg_pool3d(
                prediction, kernel_size=factor, stride=factor
            )
            half_target = F.avg_pool3d(
                target, kernel_size=factor, stride=factor
            )
            half_mask = F.avg_pool3d(
                mask, kernel_size=factor, stride=factor
            ).clamp(0.0, 1.0)
            half_anatomy = F.avg_pool3d(
                anatomy_weight, kernel_size=factor, stride=factor
            )
            half_map = self.half_ssim(
                half_prediction, half_target, return_map=True
            )
            half_error = 1.0 - half_map.clamp(-1.0, 1.0)
            half_focal = self._focal_ssim_term(
                half_error, half_mask, half_anatomy
            )
            upsampled_half_error = F.interpolate(
                half_error,
                size=full_error.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            cross_scale = self._masked_mean(
                F.smooth_l1_loss(
                    full_error,
                    upsampled_half_error,
                    reduction="none",
                ),
                anatomy_weight * mask,
            )
        else:
            half_focal = full_focal
            cross_scale = prediction.new_zeros(())

        focal_ssim = (
            self.scale_weights[0] * full_focal
            + self.scale_weights[1] * half_focal
        )
        residual_ncc = self._local_residual_ncc(
            prediction, target, atlas, mask, anatomy_weight
        )
        global_ssim_error = full_error.mean()
        brain_ssim_error = self._masked_mean(full_error, mask)
        loss = (
            self.focal_ssim_weight * focal_ssim
            + self.cross_scale_weight * cross_scale
            + self.residual_ncc_weight * residual_ncc
            + self.global_ssim_weight * global_ssim_error
            + self.brain_ssim_weight * brain_ssim_error
        )
        return loss, {
            "afmr_ssim_loss": loss.detach(),
            "focal_ssim_loss": focal_ssim.detach(),
            "full_focal_ssim_loss": full_focal.detach(),
            "half_focal_ssim_loss": half_focal.detach(),
            "cross_scale_loss": cross_scale.detach(),
            "residual_ncc_loss": residual_ncc.detach(),
            "global_ssim_loss": global_ssim_error.detach(),
            "brain_ssim_loss": brain_ssim_error.detach(),
            "anatomy_weight_mean": self._masked_mean(
                anatomy_weight, mask
            ).detach(),
        }
