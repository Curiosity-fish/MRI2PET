"""Unified 13-stage Exp027 trainer.

The trainer keeps the stochastic diffusion objective in Phase I and switches to
the model's fixed-timestep direct path for every later phase.  All region losses
are computed from the fixed Parc98 labelmap supplied by the dataset; the labelmap
is never passed to the model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp026.training.diffusion import DiffusionSchedule
from exp026.utils.checkpoints import load_checkpoint, save_checkpoint
from exp027.losses.parc98_losses import patch_boundary_loss
from exp027.losses.soft_region import (
    regional_metabolic_graph_loss_soft,
    soft_patch_to_region_alignment_loss,
)
from exp027.priors.statistical import (
    fit_residual_prior,
    load_residual_prior,
    save_residual_prior,
)
from exp027.training.stages import STAGE_NAMES, configure_trainable_modules, set_stage_mode
from exp027.utils.parc98_graph import Parc98Graph
from metrics.image_metrics import compute_image_metrics_detailed
from utils.ssim import SSIM3D


class Exp027Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
        graph: Parc98Graph,
        *,
        train_dataset: Any = None,
        rebuild_residual_prior: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.graph = graph
        self.train_dataset = train_dataset
        self.rebuild_residual_prior = bool(rebuild_residual_prior)
        self.output_root = Path(config["experiment"]["output_dir"]).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.patch_batch_size = int(config["training"]["patch_batch_size"])
        self.amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
        self.gradient_clip = float(config["training"].get("gradient_clip", 0.0))
        self.metric_data_range = float(config["training"].get("metric_data_range", 2.0))
        self.multiscale_factor = max(1, int(config["training"].get("multiscale_ssim_factor", 2)))
        self.ssim_loss = SSIM3D(
            window_size=int(config["training"].get("ssim_window_size", 11)),
            data_range=self.metric_data_range,
        ).to(device)
        diffusion = config["diffusion"]
        self.schedule = DiffusionSchedule(
            timesteps=int(diffusion["timesteps"]),
            beta_start=float(diffusion.get("beta_start", 1e-4)),
            beta_end=float(diffusion.get("beta_end", 2e-2)),
            device=device,
        )
        initial = config["experiment"].get("initial_checkpoint")
        if initial:
            load_checkpoint(initial, self.model, map_location=device, strict=True)

    @staticmethod
    def _weight(stage_config: dict, name: str, default: float = 0.0) -> float:
        return float(stage_config.get("loss", {}).get(name, default))

    def _batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = batch["image"].to(self.device, non_blocking=True).float()
        target = batch["label"].to(self.device, non_blocking=True).float()
        table = batch["table"].to(self.device, non_blocking=True).float()
        table_mask = batch.get("mask")
        if table_mask is None:
            table_mask = torch.zeros_like(table)
        else:
            table_mask = table_mask.to(self.device, non_blocking=True).float()
        return image, target, table, table_mask

    def _brain_mask(self, batch: dict, target: torch.Tensor) -> torch.Tensor:
        mask = batch.get("brain_mask")
        if mask is None:
            return torch.ones_like(target)
        return mask.to(self.device, non_blocking=True).float()

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask).sum() / mask.sum().clamp_min(1e-6)

    def _masked_l1(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._masked_mean((prediction - target).abs(), mask)

    def _gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        terms = []
        for dimension in (-3, -2, -1):
            pred_diff = torch.diff(prediction, dim=dimension)
            target_diff = torch.diff(target, dim=dimension)
            left = [slice(None)] * mask.ndim
            right = [slice(None)] * mask.ndim
            left[dimension] = slice(1, None)
            right[dimension] = slice(None, -1)
            terms.append(self._masked_l1(pred_diff, target_diff, mask[tuple(left)] * mask[tuple(right)]))
        return torch.stack(terms).mean()

    @staticmethod
    def _laplacian_pyramid_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Three-scale 3D Laplacian pyramid loss on PET patches."""
        current_prediction = prediction.reshape(-1, prediction.shape[2], *prediction.shape[-3:])
        current_target = target.reshape(-1, target.shape[2], *target.shape[-3:])
        result = current_prediction.new_zeros(())
        for level in range(2):
            down_prediction = F.avg_pool3d(current_prediction, 2, 2)
            down_target = F.avg_pool3d(current_target, 2, 2)
            up_prediction = F.interpolate(
                down_prediction, size=current_prediction.shape[-3:],
                mode="trilinear", align_corners=False,
            )
            up_target = F.interpolate(
                down_target, size=current_target.shape[-3:],
                mode="trilinear", align_corners=False,
            )
            result = result + (2.0 ** (-level)) * F.l1_loss(
                current_prediction - up_prediction,
                current_target - up_target,
            )
            current_prediction, current_target = down_prediction, down_target
        return result + 0.25 * F.l1_loss(current_prediction, current_target)

    def _target_latent(self, pet: torch.Tensor) -> torch.Tensor:
        return self.model.encode_pet_patches(
            pet,
            self.patch_batch_size,
            sample=False,
            encoder_grad=False,
        )

    def _patch_vae_step(
        self,
        pet: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self.model.extract_image_patches(pet)
        target_mask = (
            self.model.extract_image_patches(brain_mask)
            if brain_mask is not None else torch.ones_like(target)
        )
        reconstructions = []
        mus = []
        logvars = []
        for start in range(0, self.model.num_patches, self.patch_batch_size):
            end = min(start + self.patch_batch_size, self.model.num_patches)
            patch = target[:, start:end]
            batch_size, count = patch.shape[:2]
            reconstruction, mu, logvar, _ = self.model.patch_vae(
                patch.reshape(batch_size * count, patch.shape[2], *patch.shape[-3:])
            )
            reconstructions.append(reconstruction.reshape(batch_size, count, *reconstruction.shape[1:]))
            mus.append(mu)
            logvars.append(logvar)
        reconstruction = torch.cat(reconstructions, dim=1)
        mu = torch.cat(mus, dim=0)
        logvar = torch.cat(logvars, dim=0)
        rec = self._masked_mean((reconstruction - target).abs(), target_mask)
        lap = self._laplacian_pyramid_loss(reconstruction, target)
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
        loss = (
            self._weight(stage_config, "rec", 1.0) * rec
            + self._weight(stage_config, "lap", 0.1) * lap
            + self._weight(stage_config, "kl", 1e-6) * kl
        )
        metrics = {"loss": loss.detach(), "rec": rec.detach(), "lap": lap.detach(), "kl": kl.detach()}
        return loss, metrics

    def _mri_prior_step(self, mri: torch.Tensor, pet: torch.Tensor, stage_config: dict) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            target = self._target_latent(pet)
            targets = self.model.build_latent_patch_pyramid(target)
        controls = self.model.mri_control(mri)
        control_patches = self.model.build_control_patch_pyramid(controls)
        predictions = self.model.mri_prior_heads(control_patches)
        high = F.smooth_l1_loss(predictions["high"], targets["high"])
        middle = F.smooth_l1_loss(predictions["middle"], targets["middle"])
        low = F.smooth_l1_loss(predictions["low"], targets["low"])
        loss = (
            self._weight(stage_config, "high", 1.0) * high
            + self._weight(stage_config, "middle", 0.5) * middle
            + self._weight(stage_config, "low", 0.25) * low
        )
        return loss, {"loss": loss.detach(), "mri_high": high.detach(), "mri_middle": middle.detach(), "mri_low": low.detach()}

    def _diffusion_step(
        self,
        mri: torch.Tensor,
        pet: torch.Tensor,
        table: torch.Tensor,
        table_mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict]:
        clean = self._target_latent(pet)
        batch_size = clean.shape[0]
        timesteps = torch.randint(0, self.schedule.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(clean)
        noisy = self.schedule.q_sample(clean, timesteps, noise)
        predicted_noise, _ = self.model.denoise_patches(
            noisy,
            timesteps,
            mri,
            None,
            None,
            self.patch_batch_size,
            use_table=False,
            use_global=False,
            inject_table_locally=False,
        )
        alpha_bar = self.schedule.sqrt_alpha_bars[timesteps].square()
        one_minus = (1.0 - alpha_bar).clamp_min(1e-8)
        snr = alpha_bar / one_minus
        gamma = float(self.config["diffusion"].get("min_snr_gamma", 5.0))
        snr_weight = (snr.clamp(max=gamma) / snr.clamp_min(1e-8)).to(noise.dtype)
        noise_error = (predicted_noise - noise).square().flatten(1).mean(1)
        diffusion_loss = (noise_error * snr_weight).mean()
        predicted_clean = self.schedule.predict_clean(noisy, timesteps, predicted_noise)
        valid_clean = (alpha_bar >= float(self.config["diffusion"].get("clean_alpha_min", 0.05))).float()
        clean_error = F.smooth_l1_loss(predicted_clean, clean, reduction="none").flatten(1).mean(1)
        clean_loss = (clean_error * valid_clean).sum() / valid_clean.sum().clamp_min(1.0)

        controls = self.model.mri_control(mri)
        control_patches = self.model.build_control_patch_pyramid(controls)
        priors = self.model.mri_prior_heads(control_patches)
        target_pyramid = self.model.build_latent_patch_pyramid(clean)
        mri_prior_loss = (
            F.smooth_l1_loss(priors["high"], target_pyramid["high"])
            + 0.5 * F.smooth_l1_loss(priors["middle"], target_pyramid["middle"])
            + 0.25 * F.smooth_l1_loss(priors["low"], target_pyramid["low"])
        )
        loss = (
            self._weight(stage_config, "diffusion", 1.0) * diffusion_loss
            + self._weight(stage_config, "clean", 0.1) * clean_loss
            + self._weight(stage_config, "mri_prior", 0.05) * mri_prior_loss
        )
        return loss, {
            "loss": loss.detach(),
            "diffusion": diffusion_loss.detach(),
            "clean": clean_loss.detach(),
            "mri_prior": mri_prior_loss.detach(),
            "timestep": timesteps.float().mean().detach(),
        }

    @staticmethod
    def _stage_flags(stage: str) -> tuple[bool, bool, bool, bool]:
        if stage == "deterministic_prior":
            return False, False, False, False
        if stage == "deterministic_local":
            return False, False, True, False
        if stage == "table_mri_prior":
            return True, False, False, False
        if stage == "global_mamba":
            return True, True, True, False
        if stage in ("joint_finetune", "decoder_calibration", "structural_head", "structural_decoder", "spatial_scalars", "final_calibration"):
            return True, True, True, True
        raise ValueError(f"Stage {stage} has no direct reconstruction flags")

    def _image_terms(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        target = target.float()
        mask = brain_mask.float()
        l1 = F.l1_loss(prediction, target)
        mse = F.mse_loss(prediction, target)
        brain_l1 = self._masked_l1(prediction, target, mask)
        brain_mse = self._masked_mean((prediction - target).square(), mask)
        ssim_map = self.ssim_loss(prediction, target, return_map=True)
        ssim = ssim_map.mean()
        brain_ssim = self._masked_mean(ssim_map, mask)
        multiscale = prediction.new_zeros(())
        if self.multiscale_factor > 1:
            pooled_prediction = F.avg_pool3d(prediction, self.multiscale_factor, self.multiscale_factor)
            pooled_target = F.avg_pool3d(target, self.multiscale_factor, self.multiscale_factor)
            multiscale = 1.0 - self.ssim_loss(pooled_prediction, pooled_target)
        gradient = self._gradient_loss(prediction, target, mask)
        range_loss = F.relu(prediction.abs() - 1.0).mean()
        loss = (
            self._weight(stage_config, "l1", 0.75) * l1
            + self._weight(stage_config, "mse", 0.5) * mse
            + self._weight(stage_config, "brain_l1", 1.0) * brain_l1
            + self._weight(stage_config, "brain_mse", 0.25) * brain_mse
            + self._weight(stage_config, "ssim", 0.0) * (1.0 - ssim)
            + self._weight(stage_config, "brain_ssim", 0.0) * (1.0 - brain_ssim)
            + self._weight(stage_config, "multiscale_ssim", 0.0) * multiscale
            + self._weight(stage_config, "gradient", 0.1) * gradient
            + self._weight(stage_config, "range", 0.1) * range_loss
        )
        return loss, {
            "loss_image": loss.detach(),
            "l1": l1.detach(),
            "mse": mse.detach(),
            "brain_l1": brain_l1.detach(),
            "brain_mse": brain_mse.detach(),
            "ssim_loss": (1.0 - ssim).detach(),
            "brain_ssim_loss": (1.0 - brain_ssim).detach(),
            "multiscale_ssim": multiscale.detach(),
            "gradient": gradient.detach(),
            "range": range_loss.detach(),
        }

    def _region_terms(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        refined_latent: torch.Tensor,
        target_latent: torch.Tensor,
        parc98: torch.Tensor,
        brain_mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = prediction.new_zeros(())
        metrics: dict[str, torch.Tensor] = {}
        rmgc_weight = self._weight(stage_config, "rmgc", 0.0)
        if rmgc_weight != 0.0:
            rmgc, values = regional_metabolic_graph_loss_soft(
                prediction, target, parc98.long(), self.graph, brain_mask
            )
            total = total + rmgc_weight * rmgc
            metrics.update({f"region_{key}": value for key, value in values.items()})
        pra_weight = self._weight(stage_config, "pra", 0.0)
        lgr_weight = self._weight(stage_config, "lgr", 0.0)
        if pra_weight != 0.0 or lgr_weight != 0.0:
            predicted_volume = self.model.stitch_latent_patches(refined_latent)
            target_volume = self.model.stitch_latent_patches(target_latent)
            latent_total, values = soft_patch_to_region_alignment_loss(
                predicted_volume, target_volume, parc98.long(), self.graph
            )
            total = total + pra_weight * values["pra_raw"] + lgr_weight * values["lgr_raw"]
            metrics.update({
                f"latent_{key}": value
                for key, value in values.items()
                if not key.endswith("_raw")
            })
        return total, metrics

    def _direct_step(self, stage: str, batch: dict, stage_config: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mri, pet, table, table_mask = self._batch(batch)
        brain_mask = self._brain_mask(batch, pet)
        use_table, use_global, use_local, use_image = self._stage_flags(stage)
        prediction, auxiliary = self.model.predict_direct(
            mri,
            table,
            table_mask,
            self.patch_batch_size,
            use_table=use_table,
            use_global=use_global,
            use_local_refiner=use_local,
            use_image_refiner=use_image,
            clamp_output=False,
        )
        metrics: dict[str, torch.Tensor] = {}
        total = prediction.new_zeros(())
        target_latent = None
        needs_latent = any(
            self._weight(stage_config, key, 0.0) != 0.0
            for key in ("direct", "latent", "pra", "lgr", "real_image", "predicted_image", "table_res")
        )
        if needs_latent:
            with torch.no_grad():
                target_latent = self._target_latent(pet)

        if stage == "table_mri_prior":
            context = auxiliary["table_context"]
            if context is None:
                raise RuntimeError("Exp027 table stage did not produce a table context")
            mri_prior = auxiliary["mri_prior_pyramid"]["high"].detach()
            predicted_residual = context["prior_patches"] - mri_prior
            target_residual = target_latent.detach() - mri_prior
            residual_loss = F.smooth_l1_loss(predicted_residual, target_residual)
            drop_prob = float(self.config.get("table", {}).get("extra_missing_probability", 0.15))
            dropped_mask = torch.maximum(
                table_mask,
                (torch.rand_like(table_mask) < drop_prob).float(),
            )
            dropped_prediction, dropped_aux = self.model.predict_direct(
                mri,
                table,
                dropped_mask,
                self.patch_batch_size,
                use_table=True,
                use_global=False,
                use_local_refiner=False,
                use_image_refiner=False,
                clamp_output=False,
            )
            del dropped_prediction
            dropped_context = dropped_aux["table_context"]
            reliability = context["reliability"].detach()
            missing_loss = reliability.mean() * F.l1_loss(
                dropped_context["prior_patches"], context["prior_patches"].detach()
            )
            total = (
                self._weight(stage_config, "table_res", 1.0) * residual_loss
                + self._weight(stage_config, "missing", 0.1) * missing_loss
            )
            metrics.update({"table_res": residual_loss.detach(), "missing": missing_loss.detach()})
            metrics["loss"] = total.detach()
            return total, metrics

        if stage == "decoder_calibration":
            if target_latent is None:
                raise RuntimeError("Decoder calibration requires target latent")
            target_patches = self.model.extract_image_patches(pet)
            real_decoded = self.model.decode_latent_patches(target_latent.detach(), self.patch_batch_size)
            predicted_decoded = self.model.decode_latent_patches(
                auxiliary["refined_latent"].detach(), self.patch_batch_size
            )
            real_pet = self.model.stitch_image_patches(real_decoded)
            predicted_pet = self.model.stitch_image_patches(predicted_decoded)
            real_loss, real_values = self._image_terms(real_pet, pet, brain_mask, stage_config)
            pred_loss, pred_values = self._image_terms(predicted_pet, pet, brain_mask, stage_config)
            node_weight = self._weight(stage_config, "node", 0.0)
            region_loss = prediction.new_zeros(())
            if node_weight != 0.0:
                region_loss, region_values = regional_metabolic_graph_loss_soft(
                    predicted_pet, pet, batch["parc98"].to(self.device).long(), self.graph, brain_mask
                )
                metrics.update({f"decoder_{key}": value for key, value in region_values.items()})
            seam = patch_boundary_loss(predicted_pet, pet, tuple(self.model.patch_size))
            total = (
                self._weight(stage_config, "real_image", 0.3) * real_loss
                + self._weight(stage_config, "predicted_image", 0.7) * pred_loss
                + node_weight * region_loss
                + self._weight(stage_config, "seam", 0.02) * seam
            )
            metrics.update({f"real_{key}": value for key, value in real_values.items()})
            metrics.update({f"pred_{key}": value for key, value in pred_values.items()})
            metrics["seam"] = seam.detach()
            refiner_weight = self._weight(stage_config, "refiner_image", 0.0)
            if refiner_weight != 0.0:
                refiner_loss, refiner_values = self._image_terms(
                    prediction, pet, brain_mask, stage_config
                )
                total = total + refiner_weight * refiner_loss
                metrics.update({f"refiner_{key}": value for key, value in refiner_values.items()})
            metrics["loss"] = total.detach()
            return total, metrics

        image_loss, image_values = self._image_terms(prediction, pet, brain_mask, stage_config)
        total = total + self._weight(stage_config, "image", 1.0) * image_loss
        metrics.update(image_values)
        if target_latent is not None and self._weight(stage_config, "direct", 0.0) != 0.0:
            direct = F.smooth_l1_loss(auxiliary["refined_latent"].float(), target_latent.float())
            total = total + self._weight(stage_config, "direct", 1.0) * direct
            metrics["direct"] = direct.detach()
        if target_latent is not None and (
            self._weight(stage_config, "pra", 0.0) != 0.0 or self._weight(stage_config, "lgr", 0.0) != 0.0
        ):
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
        if self._weight(stage_config, "seam", 0.0) != 0.0:
            seam = patch_boundary_loss(prediction, pet, tuple(self.model.patch_size))
            total = total + self._weight(stage_config, "seam", 0.0) * seam
            metrics["seam"] = seam.detach()
        if stage == "joint_finetune" and self._weight(stage_config, "prior", 0.0) != 0.0:
            table_context = auxiliary.get("table_context")
            if table_context is None or target_latent is None:
                raise RuntimeError("Exp027 joint fine-tuning requires the table residual context")
            prior_loss = F.smooth_l1_loss(
                table_context["prior_patches"], target_latent.detach()
            )
            total = total + self._weight(stage_config, "prior", 0.0) * prior_loss
            metrics["prior"] = prior_loss.detach()
        if stage in ("deterministic_prior", "deterministic_local"):
            with torch.no_grad():
                target = self._target_latent(pet)
                target_levels = self.model.build_latent_patch_pyramid(target)
            controls = self.model.mri_control(mri)
            predicted_levels = self.model.mri_prior_heads(self.model.build_control_patch_pyramid(controls))
            prior_loss = (
                F.smooth_l1_loss(predicted_levels["high"], target_levels["high"])
                + 0.5 * F.smooth_l1_loss(predicted_levels["middle"], target_levels["middle"])
                + 0.25 * F.smooth_l1_loss(predicted_levels["low"], target_levels["low"])
            )
            total = total + self._weight(stage_config, "mri_prior", 1.0) * prior_loss
            metrics["mri_prior"] = prior_loss.detach()
        metrics["loss"] = total.detach()
        return total, metrics

    def _selection_score(self, metrics: dict[str, float]) -> float:
        weights = self.config["training"].get(
            "selection_weights",
            {"ssim": 0.55, "brain_ssim": 0.25, "rmse": 0.10, "brain_rmse": 0.10},
        )
        if "ssim" not in metrics:
            return float(metrics["loss"])
        return (
            float(weights.get("ssim", 0.55)) * (1.0 - metrics["ssim"])
            + float(weights.get("brain_ssim", 0.25)) * (1.0 - metrics["brain_ssim"])
            + float(weights.get("rmse", 0.10)) * metrics["rmse"]
            + float(weights.get("brain_rmse", 0.10)) * metrics["brain_rmse"]
        )

    def _official_metrics(self, prediction: torch.Tensor, target: torch.Tensor, brain: torch.Tensor) -> dict[str, torch.Tensor]:
        values = compute_image_metrics_detailed(
            prediction.detach().float().clamp(-1.0, 1.0),
            target.detach().float(),
            mask=brain.detach().float(),
            data_range=self.metric_data_range,
        )
        return {key: prediction.new_tensor(float(value)) for key, value in values.items()}

    def _step(self, stage: str, batch: dict, stage_config: dict, include_official: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mri, pet, table, table_mask = self._batch(batch)
        brain = self._brain_mask(batch, pet)
        if stage == "patch_vae":
            return self._patch_vae_step(pet, brain, stage_config)
        if stage == "mri_patch_prior":
            return self._mri_prior_step(mri, pet, stage_config)
        if stage == "local_diffusion":
            return self._diffusion_step(mri, pet, table, table_mask, stage_config)
        loss, metrics = self._direct_step(stage, batch, stage_config)
        if include_official:
            with torch.no_grad():
                # 官方验证指标必须在 fp32 下计算，与独立评测脚本 eval_exp030_image_metrics.py 保持一致；
                # 在 fp16/autocast 下前向会使 SSIM 产生约 0.005 的系统性偏差。
                with torch.cuda.amp.autocast(enabled=False):
                    use_table, use_global, use_local, use_image = self._stage_flags(stage)
                    prediction, _ = self.model.predict_direct(
                        mri, table, table_mask, self.patch_batch_size,
                        use_table=use_table,
                        use_global=use_global,
                        use_local_refiner=use_local,
                        use_image_refiner=use_image,
                        clamp_output=False,
                    )
                    official = self._official_metrics(prediction, pet, brain)
            metrics.update(official)
            metrics["selection_score"] = prediction.new_tensor(
                self._selection_score({key: float(value.item()) for key, value in official.items()})
            )
        return loss, metrics

    @staticmethod
    def _append_metrics(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _read_best(path: Path) -> float:
        if not path.exists():
            return float("inf")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return float(payload.get("score", float("inf")))

    @staticmethod
    def _write_best(path: Path, stage: str, epoch: int, score: Optional[float], metrics: Optional[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump({"stage": stage, "epoch": int(epoch), "score": score, "metrics": metrics}, handle, indent=2)
        temporary.replace(path)

    @staticmethod
    def _record_metrics(metrics: dict[str, float] | None) -> dict[str, float] | None:
        return None if metrics is None else {key: float(value) for key, value in metrics.items()}

    def _set_residual_prior_if_available(self, *, force_fit: bool = False) -> None:
        if not hasattr(self.model, "set_residual_prior"):
            return
        payload = None
        if not force_fit:
            payload = load_residual_prior(
                self.config,
                Path(self.config["experiment"].get("repo_root", Path.cwd())),
                self.train_dataset,
                rebuild=False,
            ) if self.train_dataset is not None else None
        if payload is None:
            sample_ids = []
            if self.train_dataset is not None and hasattr(self.train_dataset, "df"):
                column = "sample_id" if "sample_id" in self.train_dataset.df.columns else self.train_dataset.df.columns[0]
                sample_ids = [str(value) for value in self.train_dataset.df[column].tolist()]
            payload = fit_residual_prior(
                self.model,
                self.train_loader,
                self.config,
                self.device,
                sample_ids=sample_ids,
                source_checkpoint=str(self.output_root / "structural_decoder" / "checkpoints" / "best.pt"),
            )
            root = Path(self.config["experiment"].get("repo_root", Path.cwd()))
            path_value = self.config.get("data", {}).get("residual_prior_path", "data/cache/exp027_voxel_residual_train.pt")
            path = Path(path_value)
            if not path.is_absolute():
                path = root / path
            save_residual_prior(payload, path)
        self.model.set_residual_prior(payload["atlas_residual_beta"], payload["atlas_residual_bias"])
        print(f"[exp027] residual prior loaded/fitted from train split: n={payload.get('num_samples')}")

    @contextmanager
    def _validation_rng(self):
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = int(self.config["training"].get("validation_seed", 2601))
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield

    def _run_epoch(self, stage: str, stage_config: dict, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], scaler: torch.cuda.amp.GradScaler, epoch: int, training: bool) -> dict[str, float]:
        set_stage_mode(self.model, stage, training=training)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(loader, desc=f"exp027 {stage} {epoch} {'train' if training else 'val'}", leave=False)
        context = nullcontext() if training else self._validation_rng()
        with context:
            for batch in progress:
                batch_size = int(batch["image"].shape[0])
                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        loss, metrics = self._step(stage, batch, stage_config, include_official=False)
                    scaler.scale(loss).backward()
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.amp):
                            _, metrics = self._step(stage, batch, stage_config, include_official=True)
                seen += batch_size
                for name, value in metrics.items():
                    totals[name] += float(value.detach().item()) * batch_size
                progress.set_postfix(loss=float(metrics["loss"].detach().item()))
        if seen == 0:
            raise RuntimeError(f"No valid batches were produced for Exp027 stage {stage}")
        return {name: value / seen for name, value in totals.items()}

    def _epoch_zero_guard(self, stage: str, stage_config: dict, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, checkpoints: Path) -> tuple[float, dict[str, float]]:
        if self.val_loader is None:
            return float("inf"), {}
        values = self._run_epoch(stage, stage_config, self.val_loader, None, scaler, 0, False)
        record = {"epoch": 0, "train": None, "val": values, "baseline_guard": True}
        for name in ("best.pt", "best_ssim.pt", "best_psnr.pt", "best_region.pt", "last.pt"):
            save_checkpoint(checkpoints / name, self.model, optimizer, stage, 0, self.config, record)
        score = self._selection_score(values)
        self._write_best(checkpoints.parent / "best_metric.json", stage, 0, score, values)
        self._append_metrics(checkpoints.parent / "metrics.jsonl", record)
        return score, values

    def run(self, only_stage: Optional[str] = None) -> None:
        if only_stage is not None and only_stage not in STAGE_NAMES:
            raise ValueError(f"Unknown Exp027 stage: {only_stage}")
        if only_stage is not None:
            stage_index = STAGE_NAMES.index(only_stage)
            has_bootstrap = bool(
                self.config["experiment"].get("initial_checkpoint")
                or self.config["experiment"].get("warm_start_checkpoint")
            )
            if stage_index > 0 and not has_bootstrap:
                missing = []
                for previous in STAGE_NAMES[:stage_index]:
                    previous_config = self.config["stages"].get(previous, {})
                    if not previous_config.get("enabled", True) or int(previous_config.get("epochs", 0)) == 0:
                        continue
                    checkpoint = self.output_root / previous / "checkpoints" / "best.pt"
                    if not checkpoint.exists():
                        missing.append(str(checkpoint))
                if missing:
                    raise RuntimeError(
                        f"Exp027 stage {only_stage} requires previous best checkpoints: {missing}"
                    )
        for stage in STAGE_NAMES:
            if only_stage is not None and stage != only_stage:
                continue
            stage_config = self.config["stages"][stage]
            epochs = int(stage_config.get("epochs", 0))
            if not stage_config.get("enabled", True) or epochs == 0:
                print(f"[exp027] skip {stage}: enabled={stage_config.get('enabled', True)}, epochs={epochs}")
                continue
            if stage == "spatial_scalars":
                self._set_residual_prior_if_available(force_fit=self.rebuild_residual_prior)

            groups = configure_trainable_modules(self.model, stage)
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if not parameters:
                raise RuntimeError(f"Exp027 stage {stage} has no trainable parameters")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(stage_config["learning_rate"]),
                weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.config["training"].get("lr_plateau_factor", 0.5)),
                patience=int(self.config["training"].get("lr_plateau_patience", 8)),
                min_lr=float(self.config["training"].get("min_learning_rate", 1e-7)),
            )
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            stage_dir = self.output_root / stage
            checkpoints = stage_dir / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=True)
            resume = stage_config.get("resume")
            start_epoch = 1
            if resume:
                checkpoint = load_checkpoint(resume, self.model, optimizer=optimizer, map_location=self.device, strict=True)
                if checkpoint.get("stage") != stage:
                    raise ValueError(f"Resume checkpoint stage={checkpoint.get('stage')} does not match {stage}")
                start_epoch = int(checkpoint["epoch"]) + 1
                best_score = self._read_best(stage_dir / "best_metric.json")
            else:
                best_score = float("inf")
                if stage in ("structural_head", "structural_decoder", "spatial_scalars", "final_calibration"):
                    best_score, _ = self._epoch_zero_guard(stage, stage_config, optimizer, scaler, checkpoints)
            best_ssim = -float("inf")
            best_psnr = -float("inf")
            best_region = float("inf")
            print(f"[exp027] stage={stage}, epochs={epochs}, trainable_groups={groups}, trainable_params={sum(p.numel() for p in parameters)}")
            for epoch in range(start_epoch, epochs + 1):
                train_metrics = self._run_epoch(stage, stage_config, self.train_loader, optimizer, scaler, epoch, True)
                val_metrics = None
                interval = max(1, int(self.config["training"].get("val_interval", 1)))
                if self.val_loader is not None and (epoch % interval == 0 or epoch == epochs):
                    val_metrics = self._run_epoch(stage, stage_config, self.val_loader, None, scaler, epoch, False)
                record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
                self._append_metrics(stage_dir / "metrics.jsonl", record)
                monitor = val_metrics if val_metrics is not None else train_metrics
                score = self._selection_score(monitor)
                scheduler.step(score)
                if score < best_score:
                    best_score = score
                    save_checkpoint(checkpoints / "best.pt", self.model, optimizer, stage, epoch, self.config, record)
                    self._write_best(stage_dir / "best_metric.json", stage, epoch, score, monitor)
                if val_metrics is not None:
                    if val_metrics.get("ssim", -float("inf")) > best_ssim:
                        best_ssim = val_metrics["ssim"]
                        save_checkpoint(checkpoints / "best_ssim.pt", self.model, optimizer, stage, epoch, self.config, record)
                    if val_metrics.get("psnr", -float("inf")) > best_psnr:
                        best_psnr = val_metrics["psnr"]
                        save_checkpoint(checkpoints / "best_psnr.pt", self.model, optimizer, stage, epoch, self.config, record)
                    region_score = val_metrics.get("region_node", val_metrics.get("loss", float("inf")))
                    if region_score < best_region:
                        best_region = region_score
                        save_checkpoint(checkpoints / "best_region.pt", self.model, optimizer, stage, epoch, self.config, record)
                save_checkpoint(checkpoints / "last.pt", self.model, optimizer, stage, epoch, self.config, record)
                print(f"[exp027] {stage} epoch={epoch} train={train_metrics} val={val_metrics}")
            best_path = checkpoints / "best.pt"
            if not best_path.exists():
                raise FileNotFoundError(f"Exp027 stage {stage} finished without {best_path}")
            selected = load_checkpoint(best_path, self.model, map_location=self.device, strict=True)
            print(f"[exp027] stage={stage} selected best.pt from epoch={selected['epoch']}")
