from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from exp025s.models.exp025s_model import Exp025SMRI2PET


def _bounded_logit(initial: float, maximum: float) -> torch.Tensor:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    ratio = min(max(float(initial) / float(maximum), 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


class MetricResidualBlock3D(nn.Module):
    def __init__(self, channels: int, norm_groups: int) -> None:
        super().__init__()
        groups = min(int(norm_groups), int(channels))
        while channels % groups != 0:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm1(value))
        residual = self.conv1(residual)
        residual = F.silu(self.norm2(residual))
        return value + self.conv2(residual)


class MetricCalibrationHead3D(nn.Module):
    """Small zero-initialized full-resolution residual head."""

    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 8,
        depth: int = 2,
        norm_groups: int = 4,
        correction_scale: float = 0.08,
    ) -> None:
        super().__init__()
        self.correction_scale = float(correction_scale)
        self.input = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            MetricResidualBlock3D(channels, norm_groups) for _ in range(int(depth))
        )
        self.output = nn.Conv3d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value)
        for block in self.blocks:
            hidden = block(hidden)
        return self.correction_scale * torch.tanh(self.output(F.silu(hidden)))


class Exp025TMRI2PET(Exp025SMRI2PET):
    """Exp025s with bounded structural and atlas-aware output calibration."""

    def __init__(
        self,
        *,
        smooth_weight: float = 0.35,
        smooth_weight_max: float = 0.60,
        atlas_weight: float = 0.05,
        atlas_weight_max: float = 0.20,
        metric_head_channels: int = 8,
        metric_head_depth: int = 2,
        metric_head_correction_scale: float = 0.08,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.smooth_weight_max = float(smooth_weight_max)
        self.atlas_weight_max = float(atlas_weight_max)
        self.smooth_weight_logit = nn.Parameter(
            _bounded_logit(smooth_weight, self.smooth_weight_max)
        )
        self.atlas_weight_logit = nn.Parameter(
            _bounded_logit(atlas_weight, self.atlas_weight_max)
        )
        self.metric_head = MetricCalibrationHead3D(
            in_channels=4,
            channels=int(metric_head_channels),
            depth=int(metric_head_depth),
            norm_groups=int(kwargs.get("norm_groups", 8)),
            correction_scale=float(metric_head_correction_scale),
        )

    @property
    def smooth_weight(self) -> torch.Tensor:
        return self.smooth_weight_max * torch.sigmoid(self.smooth_weight_logit)

    @property
    def atlas_weight(self) -> torch.Tensor:
        return self.atlas_weight_max * torch.sigmoid(self.atlas_weight_logit)

    def calibrate_prediction(
        self,
        prediction: torch.Tensor,
        mri: torch.Tensor,
        atlas_pet: torch.Tensor,
        foreground: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction32 = prediction.float()
        local_mean = F.avg_pool3d(
            prediction32,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        )
        local_mean = local_mean * foreground.float()
        smooth_weight = self.smooth_weight.to(dtype=prediction32.dtype)
        smoothed = prediction32 + smooth_weight * (local_mean - prediction32)

        atlas_weight = self.atlas_weight.to(dtype=prediction32.dtype)
        atlas_blend = smoothed + atlas_weight * (atlas_pet.float() - smoothed)
        contrast = prediction32 - local_mean
        head_input = torch.cat(
            [atlas_blend, mri.float(), atlas_pet.float(), contrast], dim=1
        )
        correction = self.metric_head(head_input)
        calibrated = (atlas_blend + correction) * foreground.float()
        return calibrated, {
            "precalibration": prediction32,
            "local_mean": local_mean,
            "atlas_blend": atlas_blend,
            "metric_correction": correction,
            "smooth_weight": smooth_weight,
            "atlas_weight": atlas_weight,
        }

    def predict_direct(
        self,
        mri: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
        patch_batch_size: int,
        *,
        use_table: bool,
        use_global: bool,
        use_local_refiner: bool,
        use_image_refiner: bool,
        clamp_output: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        prediction, auxiliary = super().predict_direct(
            mri,
            table,
            table_mask,
            patch_batch_size,
            use_table=use_table,
            use_global=use_global,
            use_local_refiner=use_local_refiner,
            use_image_refiner=use_image_refiner,
            clamp_output=False,
        )
        calibration = None
        if use_image_refiner:
            prediction, calibration = self.calibrate_prediction(
                prediction,
                mri,
                auxiliary["atlas_pet"],
                auxiliary["foreground"],
            )
        if clamp_output:
            prediction = prediction.clamp(-1.0, 1.0)
        auxiliary["metric_calibration"] = calibration
        return prediction, auxiliary


def build_exp025t_model(
    config: dict,
    table_num_features: int,
    atlas_prior: dict,
    feature_type_ids: Optional[Sequence[int]] = None,
) -> Exp025TMRI2PET:
    data = config["data"]
    model = config["model"]
    mamba = model["global_mamba"]
    table = model["table"]
    direct = model.get("direct_path", {})
    refiner = model.get("image_refiner", {})
    calibration = model.get("metric_calibration", {})
    return Exp025TMRI2PET(
        image_size=tuple(data["image_size"]),
        patch_size=tuple(data["patch_size"]),
        table_num_features=table_num_features,
        feature_type_ids=feature_type_ids,
        image_channels=int(model.get("image_channels", 1)),
        latent_channels=int(model.get("latent_channels", 4)),
        vae_base_channels=int(model.get("vae_base_channels", 16)),
        mri_channels=tuple(model.get("mri_channels", [8, 16, 32, 64])),
        local_channels=tuple(model.get("local_channels", [32, 64, 96])),
        time_dim=int(model.get("time_dim", 128)),
        table_token_dim=int(table.get("token_dim", 128)),
        table_token_grid=tuple(table.get("token_grid", [4, 4, 4])),
        table_depth=int(table.get("depth", 2)),
        table_heads=int(table.get("heads", 4)),
        num_feature_types=int(table.get("num_feature_types", 1)),
        global_token_dim=int(mamba.get("token_dim", 128)),
        mamba_depth=int(mamba.get("depth", 2)),
        mamba_state_dim=int(mamba.get("state_dim", 16)),
        mamba_conv_width=int(mamba.get("conv_width", 4)),
        mamba_expand=int(mamba.get("expand", 2)),
        mamba_backend=str(mamba.get("backend", "mamba")),
        allow_test_backend=bool(mamba.get("allow_test_backend", False)),
        norm_groups=int(model.get("norm_groups", 8)),
        dropout=float(model.get("dropout", 0.0)),
        gradient_checkpointing=bool(
            config["training"].get("gradient_checkpointing", True)
        ),
        atlas_slope=atlas_prior["slope"],
        atlas_bias=atlas_prior["bias"],
        direct_timestep=int(direct.get("timestep", 0)),
        direct_residual_scale=float(direct.get("residual_scale", 0.1)),
        coarse_latent_weight=float(direct.get("coarse_latent_weight", 0.5)),
        refiner_base_channels=int(refiner.get("base_channels", 12)),
        refiner_channel_mults=tuple(refiner.get("channel_mults", [1, 2, 4, 4])),
        refiner_blocks_per_level=int(refiner.get("blocks_per_level", 1)),
        refiner_dropout=float(refiner.get("dropout", 0.0)),
        refiner_correction_scale=float(refiner.get("correction_scale", 0.35)),
        smooth_weight=float(calibration.get("smooth_weight", 0.35)),
        smooth_weight_max=float(calibration.get("smooth_weight_max", 0.60)),
        atlas_weight=float(calibration.get("atlas_weight", 0.05)),
        atlas_weight_max=float(calibration.get("atlas_weight_max", 0.20)),
        metric_head_channels=int(calibration.get("channels", 8)),
        metric_head_depth=int(calibration.get("depth", 2)),
        metric_head_correction_scale=float(
            calibration.get("correction_scale", 0.08)
        ),
    )
