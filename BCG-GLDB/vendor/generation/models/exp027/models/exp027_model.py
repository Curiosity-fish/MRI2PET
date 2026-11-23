from __future__ import annotations

from typing import Optional, Sequence

import torch

from exp025u.models.exp025u_model import Exp025UMRI2PET


class Exp027MRI2PET(Exp025UMRI2PET):
    """Unified Exp026/025s/025t/025u backbone with Exp027 prior updates."""

    def set_residual_prior(
        self, beta: torch.Tensor, bias: torch.Tensor
    ) -> None:
        expected = (1, 1, *self.image_size)
        beta_tensor = self._atlas_tensor(beta, expected, "atlas_residual_beta")
        bias_tensor = self._atlas_tensor(bias, expected, "atlas_residual_bias")
        self.atlas_residual_beta = beta_tensor.to(
            device=self.atlas_residual_beta.device,
            dtype=self.atlas_residual_beta.dtype,
        )
        self.atlas_residual_bias = bias_tensor.to(
            device=self.atlas_residual_bias.device,
            dtype=self.atlas_residual_bias.dtype,
        )


def build_exp027_model(
    config: dict,
    table_num_features: int,
    atlas_prior: dict,
    residual_prior: Optional[dict] = None,
    feature_type_ids: Optional[Sequence[int]] = None,
) -> Exp027MRI2PET:
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
    return Exp027MRI2PET(
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
    )
