"""Exp030 model: Exp027 backbone plus an image-domain performance refiner.

The refiner is a bounded-correction 3D UNet placed after the full Exp027/029
direct path (image_refiner + spatial calibration).  Its final convolution is
zero-initialised, so with untrained weights it acts as an identity map and can
never corrupt a warm-started checkpoint.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from exp027.models.exp027_model import Exp027MRI2PET
from models.global_fusion_refiner3d import GlobalFusionRefiner3D


class Exp030MRI2PET(Exp027MRI2PET):
    def __init__(
        self,
        *,
        performance_refiner: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        refiner = performance_refiner or {}
        self.performance_refiner = GlobalFusionRefiner3D(
            in_channels=int(refiner.get("in_channels", 3)),
            out_channels=int(refiner.get("out_channels", 1)),
            base_channels=int(refiner.get("base_channels", 8)),
            channel_mults=tuple(refiner.get("channel_mults", [1, 2, 4])),
            blocks_per_level=int(refiner.get("blocks_per_level", 1)),
            norm_groups=int(refiner.get("norm_groups", 8)),
            dropout=float(refiner.get("dropout", 0.0)),
            correction_scale=float(refiner.get("correction_scale", 0.25)),
            use_checkpoint=bool(refiner.get("use_checkpoint", True)),
        )

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
        use_performance_refiner: bool = False,
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
        performance = None
        if use_performance_refiner:
            foreground = auxiliary.get("foreground")
            if foreground is None:
                foreground = torch.ones_like(prediction)
            output = self.performance_refiner(
                prediction,
                mri,
                foreground,
                return_dict=True,
            )
            performance = output
            prediction = output["refined"]
        if clamp_output:
            prediction = prediction.clamp(-1.0, 1.0)
        auxiliary["performance_refiner"] = performance
        return prediction, auxiliary


def build_exp030_model(
    config: dict,
    table_num_features: int,
    atlas_prior: dict,
    residual_prior: Optional[dict] = None,
    feature_type_ids: Optional[Sequence[int]] = None,
) -> Exp030MRI2PET:
    data = config["data"]
    model_config = config["model"]
    mamba = model_config["global_mamba"]
    table = model_config["table"]
    direct = model_config.get("direct_path", {})
    refiner = model_config.get("image_refiner", {})
    calibration = model_config.get("metric_calibration", {})
    spatial = model_config.get("spatial_calibration", {})
    image_size = tuple(int(value) for value in data["image_size"])
    if residual_prior is None:
        residual_prior = {
            "atlas_beta": torch.zeros((1, 1, *image_size)),
            "atlas_bias": torch.zeros((1, 1, *image_size)),
        }
    return Exp030MRI2PET(
        image_size=image_size,
        patch_size=tuple(int(value) for value in data["patch_size"]),
        table_num_features=table_num_features,
        feature_type_ids=feature_type_ids,
        image_channels=int(model_config.get("image_channels", 1)),
        latent_channels=int(model_config.get("latent_channels", 4)),
        vae_base_channels=int(model_config.get("vae_base_channels", 16)),
        mri_channels=tuple(model_config.get("mri_channels", [8, 16, 32, 64])),
        local_channels=tuple(model_config.get("local_channels", [32, 64, 96])),
        time_dim=int(model_config.get("time_dim", 128)),
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
        norm_groups=int(model_config.get("norm_groups", 8)),
        dropout=float(model_config.get("dropout", 0.0)),
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
        atlas_residual_beta=residual_prior["atlas_beta"],
        atlas_residual_bias=residual_prior["atlas_bias"],
        post_atlas_weight=float(spatial.get("atlas_weight", 0.075)),
        post_atlas_weight_max=float(spatial.get("atlas_weight_max", 0.20)),
        residual_prior_strength=float(
            spatial.get("residual_prior_strength", 0.25)
        ),
        residual_prior_strength_max=float(
            spatial.get("residual_prior_strength_max", 0.50)
        ),
        post_smooth_weight=float(spatial.get("smooth_weight", 0.0)),
        post_smooth_weight_max=float(spatial.get("smooth_weight_max", 0.20)),
        performance_refiner=model_config.get("performance_refiner", {}),
    )


__all__ = ["Exp030MRI2PET", "build_exp030_model"]
