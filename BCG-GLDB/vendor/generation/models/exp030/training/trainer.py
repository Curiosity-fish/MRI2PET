"""Exp030 trainer: Exp029 + image-domain performance refiner + frequency fidelity.

The only enabled stage is `joint_finetune`.  The trainer:
  * runs the full Exp029 joint direct step but with the new `performance_refiner`
    applied after the base direct path (use_performance_refiner=True);
  * adds `frequency_l1_loss` (3D FFT magnitude L1) to the image terms;
  * keeps Exp029's SSIM-primary selection formula so the bootstrap incumbent
    score from Exp029's best.pt is directly comparable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exp029.losses.pet_fidelity import local_contrast_l1_loss
from exp029.training.trainer import Exp029Trainer
from exp030.losses.pet_fidelity import frequency_l1_loss


class Exp030Trainer(Exp029Trainer):
    def _image_terms(self, prediction, target, brain_mask, stage_config):
        total, values = super()._image_terms(
            prediction, target, brain_mask, stage_config
        )
        weight = self._weight(stage_config, "frequency", 0.0)
        if weight == 0.0:
            return total, values
        frequency = frequency_l1_loss(
            prediction,
            target,
            log_scale=bool(self.config["training"].get("frequency_log_scale", True)),
            high_freq_boost=float(self.config["training"].get("frequency_high_boost", 1.0)),
        )
        total = total + weight * frequency
        values["frequency"] = frequency.detach()
        values["loss_image"] = total.detach()
        return total, values

    def _joint_direct_step(self, batch: dict, stage_config: dict):
        """Joint reconstruction with performance refiner and frequency fidelity."""

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
            use_performance_refiner=True,
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
            raise RuntimeError("Exp030 joint stage requires table residual context")
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
                            use_performance_refiner=True,
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


__all__ = ["Exp030Trainer"]
