"""Exp029 reconstruction-focused trainer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from exp028.training.trainer import Exp028Trainer
from exp029.losses.pet_fidelity import local_contrast_l1_loss


class Exp029Trainer(Exp028Trainer):
    """Improve reconstruction while preserving the Exp027/Exp028 interfaces."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_stage: Optional[str] = None
        self._active_epoch = 0

    def _run_epoch(self, stage, stage_config, loader, optimizer, scaler, epoch, training):
        self._active_stage = stage
        self._active_epoch = int(epoch)
        return super()._run_epoch(stage, stage_config, loader, optimizer, scaler, epoch, training)

    def _image_terms(self, prediction, target, brain_mask, stage_config):
        total, values = super()._image_terms(
            prediction, target, brain_mask, stage_config
        )
        weight = self._weight(stage_config, "local_contrast", 0.0)
        if weight == 0.0:
            return total, values
        warmup = max(
            1,
            int(self.config["training"].get("local_contrast_warmup_epochs", 12)),
        )
        ramp = min(1.0, max(0.0, float(self._active_epoch - 1) / float(warmup)))
        contrast = local_contrast_l1_loss(
            prediction,
            target,
            kernel_size=int(self.config["training"].get("local_contrast_kernel", 3)),
        )
        total = total + weight * ramp * contrast
        values["local_contrast"] = contrast.detach()
        values["loss_image"] = total.detach()
        return total, values

    def _selection_score(self, metrics: dict[str, float]) -> float:
        if "stage1_ssim" in metrics:
            return 1.0 - float(metrics["stage1_ssim"])
        if "ssim" not in metrics:
            return float(metrics["loss"])
        weights = self.config["training"].get(
            "selection_weights",
            {"ssim": 0.84, "brain_ssim": 0.10, "rmse": 0.03, "brain_rmse": 0.02, "mae": 0.01},
        )
        return (
            float(weights.get("ssim", 0.84)) * (1.0 - float(metrics["ssim"]))
            + float(weights.get("brain_ssim", 0.10)) * (1.0 - float(metrics["brain_ssim"]))
            + float(weights.get("rmse", 0.03)) * float(metrics["rmse"])
            + float(weights.get("brain_rmse", 0.02)) * float(metrics["brain_rmse"])
            + float(weights.get("mae", 0.01)) * float(metrics["mae"])
        )

    def _joint_direct_step(self, batch: dict, stage_config: dict):
        """Joint reconstruction with structure-function residual supervision."""

        mri, pet, table, table_mask = self._batch(batch)
        brain_mask = self._brain_mask(batch, pet)
        prediction, auxiliary = self.model.predict_direct(
            mri,
            table,
            table_mask,
            self.patch_batch_size,
            use_table=True,
            use_global=True,
            use_local_refiner=True,
            use_image_refiner=True,
            clamp_output=False,
        )
        with torch.no_grad():
            target_latent = self._target_latent(pet)

        image_loss, image_values = self._image_terms(
            prediction, pet, brain_mask, stage_config
        )
        total = self._weight(stage_config, "image", 1.0) * image_loss
        metrics = dict(image_values)

        direct = F.smooth_l1_loss(
            auxiliary["refined_latent"].float(), target_latent.float()
        )
        total = total + self._weight(stage_config, "direct", 1.0) * direct
        metrics["direct"] = direct.detach()

        parc98 = batch["parc98"].to(self.device, non_blocking=True)
        region_total, region_values = self._region_terms(
            prediction,
            pet,
            auxiliary["refined_latent"],
            target_latent,
            parc98,
            brain_mask,
            stage_config,
        )
        total = total + region_total
        metrics.update(region_values)

        seam_weight = self._weight(stage_config, "seam", 0.0)
        if seam_weight != 0.0:
            from exp027.losses.parc98_losses import patch_boundary_loss

            seam = patch_boundary_loss(prediction, pet, tuple(self.model.patch_size))
            total = total + seam_weight * seam
            metrics["seam"] = seam.detach()

        table_context = auxiliary.get("table_context")
        if table_context is None:
            raise RuntimeError("Exp029 joint stage requires table residual context")
        mri_prior = auxiliary["mri_prior_pyramid"]["high"].detach()
        predicted_residual = table_context["prior_patches"] - mri_prior
        target_residual = target_latent.detach() - mri_prior
        residual_prior = F.smooth_l1_loss(predicted_residual, target_residual)
        total = total + self._weight(stage_config, "prior", 0.0) * residual_prior
        metrics["prior_residual"] = residual_prior.detach()
        metrics["loss"] = total.detach()
        return total, metrics

    def _step(self, stage, batch, stage_config, include_official):
        if stage != "joint_finetune":
            return super()._step(stage, batch, stage_config, include_official)

        self._current_mri = batch["image"].to(self.device, non_blocking=True).float()
        try:
            loss, metrics = self._joint_direct_step(batch, stage_config)
            if include_official:
                mri, pet, table, table_mask = self._batch(batch)
                brain = self._brain_mask(batch, pet)
                with torch.no_grad():
                    # 官方验证指标必须在 fp32 下计算，与独立评测脚本 eval_exp030_image_metrics.py 保持一致；
                    # 在 fp16/autocast 下前向会使 SSIM 产生约 0.005 的系统性偏差。
                    with torch.cuda.amp.autocast(enabled=False):
                        prediction, _ = self.model.predict_direct(
                            mri,
                            table,
                            table_mask,
                            self.patch_batch_size,
                            use_table=True,
                            use_global=True,
                            use_local_refiner=True,
                            use_image_refiner=True,
                            clamp_output=False,
                        )
                        official = self._official_metrics(prediction, pet, brain)
                metrics.update(official)
                metrics["selection_score"] = prediction.new_tensor(
                    self._selection_score({k: float(v.item()) for k, v in official.items()})
                )
            return loss, metrics
        finally:
            self._current_mri = None


__all__ = ["Exp029Trainer"]
