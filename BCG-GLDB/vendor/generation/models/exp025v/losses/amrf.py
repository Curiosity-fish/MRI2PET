from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from utils.ssim import SSIM3D


class AMRFLoss(torch.nn.Module):
    """Anatomy-guided metabolic regional fidelity loss for MRI-to-PET."""

    def __init__(
        self,
        *,
        window_size: int = 11,
        data_range: float = 2.0,
        region_grid: Sequence[int] = (4, 4, 4),
        residual_downsample: int = 2,
        residual_window: int = 5,
        char_eps: float = 1e-3,
        anatomy_strength: float = 1.0,
        anatomy_clip: float = 4.0,
        uptake_temperature: float = 0.25,
        residual_saliency_floor: float = 0.25,
        global_char_weight: float = 0.50,
        brain_char_weight: float = 1.00,
        mse_weight: float = 0.50,
        brain_mse_weight: float = 0.50,
        anatomy_edge_weight: float = 0.25,
        regional_mean_weight: float = 0.75,
        regional_std_weight: float = 0.25,
        spatial_js_weight: float = 0.10,
        residual_ncc_weight: float = 0.20,
        global_ssim_weight: float = 0.35,
        brain_ssim_weight: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(region_grid) != 3 or any(int(v) <= 0 for v in region_grid):
            raise ValueError("region_grid must contain three positive values")
        if residual_window % 2 == 0:
            raise ValueError("residual_window must be odd")
        self.ssim = SSIM3D(
            window_size=int(window_size), data_range=float(data_range)
        )
        self.region_grid = tuple(int(v) for v in region_grid)
        self.residual_downsample = max(1, int(residual_downsample))
        self.residual_window = int(residual_window)
        self.char_eps = float(char_eps)
        self.anatomy_strength = float(anatomy_strength)
        self.anatomy_clip = float(anatomy_clip)
        self.uptake_temperature = float(uptake_temperature)
        self.residual_saliency_floor = float(residual_saliency_floor)
        self.global_char_weight = float(global_char_weight)
        self.brain_char_weight = float(brain_char_weight)
        self.mse_weight = float(mse_weight)
        self.brain_mse_weight = float(brain_mse_weight)
        self.anatomy_edge_weight = float(anatomy_edge_weight)
        self.regional_mean_weight = float(regional_mean_weight)
        self.regional_std_weight = float(regional_std_weight)
        self.spatial_js_weight = float(spatial_js_weight)
        self.residual_ncc_weight = float(residual_ncc_weight)
        self.global_ssim_weight = float(global_ssim_weight)
        self.brain_ssim_weight = float(brain_ssim_weight)
        self.eps = float(eps)

    def _weighted_mean(
        self, value: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        dimensions = tuple(range(1, value.ndim))
        numerator = (value * weight).sum(dim=dimensions)
        denominator = weight.sum(dim=dimensions).clamp_min(self.eps)
        return (numerator / denominator).mean()

    def _normalize_weight(
        self, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        dimensions = tuple(range(1, value.ndim))
        mean = (value * mask).sum(dim=dimensions, keepdim=True)
        mean = mean / mask.sum(dim=dimensions, keepdim=True).clamp_min(
            self.eps
        )
        return (value / mean.clamp_min(self.eps)).clamp(
            min=0.0, max=self.anatomy_clip
        )

    def _charbonnier(self, difference: torch.Tensor) -> torch.Tensor:
        return (difference.square() + self.char_eps**2).sqrt()

    def _anatomy_guided_edge_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mri: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        terms = []
        anatomy_means = []
        for dimension in (-3, -2, -1):
            prediction_edge = torch.diff(prediction, dim=dimension)
            target_edge = torch.diff(target, dim=dimension)
            mri_edge = torch.diff(mri, dim=dimension).abs()
            left = [slice(None)] * mask.ndim
            right = [slice(None)] * mask.ndim
            left[dimension] = slice(1, None)
            right[dimension] = slice(None, -1)
            edge_mask = mask[tuple(left)] * mask[tuple(right)]
            anatomy = self._normalize_weight(mri_edge, edge_mask).detach()
            weight = edge_mask * (1.0 + self.anatomy_strength * anatomy)
            terms.append(
                self._weighted_mean(
                    self._charbonnier(prediction_edge - target_edge),
                    weight,
                )
            )
            anatomy_means.append(self._weighted_mean(anatomy, edge_mask))
        return torch.stack(terms).mean(), torch.stack(anatomy_means).mean()

    def _regional_statistics(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        occupancy = F.adaptive_avg_pool3d(mask, self.region_grid)
        first = F.adaptive_avg_pool3d(image * mask, self.region_grid)
        second = F.adaptive_avg_pool3d(image.square() * mask, self.region_grid)
        denominator = occupancy.clamp_min(self.eps)
        mean = first / denominator
        variance = (second / denominator - mean.square()).clamp_min(self.eps)
        valid = (occupancy > 0.01).to(dtype=image.dtype)
        return mean, variance.sqrt(), occupancy * valid

    def _spatial_uptake_js(
        self,
        prediction_mean: torch.Tensor,
        target_mean: torch.Tensor,
        region_weight: torch.Tensor,
    ) -> torch.Tensor:
        valid = region_weight.flatten(1) > 0
        prediction_logits = prediction_mean.flatten(1) / max(
            self.uptake_temperature, self.eps
        )
        target_logits = target_mean.flatten(1) / max(
            self.uptake_temperature, self.eps
        )
        prediction_logits = prediction_logits.masked_fill(~valid, -1e4)
        target_logits = target_logits.masked_fill(~valid, -1e4)
        prediction_distribution = F.softmax(prediction_logits, dim=1)
        target_distribution = F.softmax(target_logits, dim=1)
        midpoint = 0.5 * (
            prediction_distribution + target_distribution
        )
        prediction_kl = (
            prediction_distribution
            * (
                prediction_distribution.clamp_min(self.eps).log()
                - midpoint.clamp_min(self.eps).log()
            )
        ).sum(dim=1)
        target_kl = (
            target_distribution
            * (
                target_distribution.clamp_min(self.eps).log()
                - midpoint.clamp_min(self.eps).log()
            )
        ).sum(dim=1)
        return (0.5 * (prediction_kl + target_kl)).mean()

    def _atlas_residual_ncc(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        atlas: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        factor = self.residual_downsample
        if factor > 1:
            prediction = F.avg_pool3d(prediction, factor, factor)
            target = F.avg_pool3d(target, factor, factor)
            atlas = F.avg_pool3d(atlas, factor, factor)
            mask = F.avg_pool3d(mask, factor, factor).clamp(0.0, 1.0)
        prediction_residual = prediction - atlas
        target_residual = target - atlas
        window = self.residual_window
        padding = window // 2
        prediction_mean = F.avg_pool3d(
            prediction_residual, window, stride=1, padding=padding
        )
        target_mean = F.avg_pool3d(
            target_residual, window, stride=1, padding=padding
        )
        cross = F.avg_pool3d(
            prediction_residual * target_residual,
            window,
            stride=1,
            padding=padding,
        )
        prediction_second = F.avg_pool3d(
            prediction_residual.square(), window, stride=1, padding=padding
        )
        target_second = F.avg_pool3d(
            target_residual.square(), window, stride=1, padding=padding
        )
        covariance = cross - prediction_mean * target_mean
        prediction_variance = (
            prediction_second - prediction_mean.square()
        ).clamp_min(0.0)
        target_variance = (
            target_second - target_mean.square()
        ).clamp_min(0.0)
        ncc = covariance / (
            prediction_variance * target_variance + self.eps
        ).sqrt()
        residual_std = target_variance.detach().sqrt()
        residual_saliency = self._normalize_weight(
            residual_std, mask
        ).detach()
        weight = mask * (
            self.residual_saliency_floor + residual_saliency
        )
        return self._weighted_mean(
            0.5 * (1.0 - ncc.clamp(-1.0, 1.0)), weight
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mri: torch.Tensor,
        mask: torch.Tensor,
        atlas: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        target = target.float()
        mri = mri.float()
        mask = mask.float()
        atlas = atlas.float()

        difference = prediction - target
        global_char = self._charbonnier(difference).mean()
        brain_char = self._weighted_mean(
            self._charbonnier(difference), mask
        )
        mse = difference.square().mean()
        brain_mse = self._weighted_mean(difference.square(), mask)
        anatomy_edge, anatomy_weight_mean = self._anatomy_guided_edge_loss(
            prediction, target, mri, mask
        )

        prediction_mean, prediction_std, region_weight = (
            self._regional_statistics(prediction, mask)
        )
        target_mean, target_std, _ = self._regional_statistics(target, mask)
        regional_mean = self._weighted_mean(
            F.smooth_l1_loss(
                prediction_mean, target_mean, reduction="none"
            ),
            region_weight,
        )
        regional_std = self._weighted_mean(
            F.smooth_l1_loss(
                prediction_std, target_std, reduction="none"
            ),
            region_weight,
        )
        spatial_js = self._spatial_uptake_js(
            prediction_mean, target_mean, region_weight
        )
        residual_ncc = self._atlas_residual_ncc(
            prediction, target, atlas, mask
        )

        ssim_map = self.ssim(prediction, target, return_map=True)
        global_ssim = 1.0 - ssim_map.mean()
        brain_ssim = 1.0 - self._weighted_mean(ssim_map, mask)

        loss = (
            self.global_char_weight * global_char
            + self.brain_char_weight * brain_char
            + self.mse_weight * mse
            + self.brain_mse_weight * brain_mse
            + self.anatomy_edge_weight * anatomy_edge
            + self.regional_mean_weight * regional_mean
            + self.regional_std_weight * regional_std
            + self.spatial_js_weight * spatial_js
            + self.residual_ncc_weight * residual_ncc
            + self.global_ssim_weight * global_ssim
            + self.brain_ssim_weight * brain_ssim
        )
        return loss, {
            "amrf_loss": loss.detach(),
            "global_char_loss": global_char.detach(),
            "brain_char_loss": brain_char.detach(),
            "mse": mse.detach(),
            "brain_mse": brain_mse.detach(),
            "anatomy_edge_loss": anatomy_edge.detach(),
            "anatomy_weight_mean": anatomy_weight_mean.detach(),
            "regional_mean_loss": regional_mean.detach(),
            "regional_std_loss": regional_std.detach(),
            "spatial_uptake_js_loss": spatial_js.detach(),
            "atlas_residual_ncc_loss": residual_ncc.detach(),
            "global_ssim_loss": global_ssim.detach(),
            "brain_ssim_loss": brain_ssim.detach(),
        }
