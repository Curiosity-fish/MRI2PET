"""Exp031_HMRD 4-stage trainer.

Stages
------
1. residual_patch_vae  : train ``residual_vae`` to reconstruct Y_res patches.
2. metabolic_prior     : train MRI control + GlobalMetabolicPrior3D on the
                         coarse low-frequency PET and the Parc98 ROI uptake
                         (targets computed from the *original* GT PET).
3. residual_diffusion  : train the global/local diffusion on the residual
                         latent (residual VAE encoder + metabolic prior frozen).
4. joint_finetune      : train everything except the residual VAE encoder;
                         selection uses the exact Exp029 selection score.

The official image metrics always use ``compute_image_metrics_detailed``
(clamp(-1,1), brain mask, data_range 2.0) under fp32, matching the independent
eval script - the protocol that was unified in the Exp029/030 audit.
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
from exp031.losses.hmrd_losses import (
    laplacian_pyramid_loss,
    multiscale_ssim_loss,
    pearson_correlation,
    roi_loss,
)
from exp031.losses.roi import extract_region_uptake
from exp031.models.decomposition import decompose_pet
from exp031.training.stages import (
    STAGE_NAMES,
    configure_trainable_modules,
    set_stage_mode,
)
from metrics.image_metrics import compute_image_metrics_detailed
from utils.ssim import SSIM3D


class Exp031Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_root = Path(config["experiment"]["output_dir"]).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.patch_batch_size = int(config["training"]["patch_batch_size"])
        self.amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
        self.gradient_clip = float(config["training"].get("gradient_clip", 0.0))
        self.metric_data_range = float(config["training"].get("metric_data_range", 2.0))
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
        self.min_snr_gamma = float(diffusion.get("min_snr_gamma", 5.0))
        self.clean_alpha_min = float(diffusion.get("clean_alpha_min", 0.05))

        loss = config.get("loss", {})
        self.lambda_diff = float(loss.get("lambda_diff", 1.0))
        self.lambda_clean = float(loss.get("lambda_clean", 0.1))
        self.lambda_coarse = float(loss.get("lambda_coarse", 0.5))
        self.lambda_roi = float(loss.get("lambda_roi", 0.15))
        self.lambda_residual = float(loss.get("lambda_residual", 0.5))
        self.lambda_image = float(loss.get("lambda_image", 1.0))
        self.lambda_lap = float(loss.get("lambda_lap", 0.1))
        self.roi_corr_weight = float(loss.get("roi_corr_weight", 0.2))
        self.lap_levels = int(loss.get("lap_levels", 3))
        self.lap_weights = tuple(float(v) for v in loss.get("lap_weights", [1.0, 0.5, 0.25]))
        self.ms_ssim_levels = max(1, int(loss.get("ms_ssim_levels", 1)))
        self.num_regions = int(config["model"].get("hmrd", {}).get("num_regions", 98))

        ablation = config["model"].get("ablation", {})
        self.use_metabolic_prior = bool(ablation.get("use_metabolic_prior", True))
        self.use_roi_tokens = bool(ablation.get("use_roi_tokens", True))
        self.use_anatomical_routing = bool(ablation.get("use_anatomical_routing", True))
        self.use_laplacian_loss = bool(ablation.get("use_laplacian_loss", True))
        self.use_table_condition = bool(ablation.get("use_table_condition", True))
        if (self.use_roi_tokens or self.use_anatomical_routing) and not self.use_metabolic_prior:
            raise ValueError(
                "Exp031 config: use_roi_tokens / use_anatomical_routing require use_metabolic_prior=true"
            )

        eval_cfg = config.get("evaluation", {})
        self.ddim_steps = int(eval_cfg.get("ddim_steps", 50))
        self.ddim_eta = float(eval_cfg.get("ddim_eta", 0.0))
        self.validation_seed = int(config["training"].get("validation_seed", 2601))

        initial = config["experiment"].get("initial_checkpoint")
        if initial:
            load_checkpoint(initial, self.model, map_location=device, strict=True)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    def _batch(self, batch: dict) -> dict:
        output = {
            "mri": batch["image"].to(self.device, non_blocking=True).float(),
            "pet": batch["label"].to(self.device, non_blocking=True).float(),
            "table": batch["table"].to(self.device, non_blocking=True).float(),
        }
        table_mask = batch.get("mask")
        if table_mask is None:
            output["table_mask"] = torch.zeros_like(output["table"])
        else:
            output["table_mask"] = table_mask.to(self.device, non_blocking=True).float()
        brain = batch.get("brain_mask")
        output["brain"] = (
            brain.to(self.device, non_blocking=True).float()
            if brain is not None
            else torch.ones_like(output["pet"])
        )
        parc98 = batch.get("parc98")
        output["parc98"] = (
            parc98.to(self.device, non_blocking=True).long()
            if parc98 is not None
            else None
        )
        return output

    @staticmethod
    def _weight(stage_config: dict, name: str, default: float = 0.0) -> float:
        return float(stage_config.get("loss", {}).get(name, default))

    def _gt_region(self, pet: torch.Tensor, parc98: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if parc98 is None:
            raise RuntimeError("Exp031 ROI loss requires the Parc98 labelmap from the dataset")
        return extract_region_uptake(pet, parc98, num_regions=self.num_regions)

    # ------------------------------------------------------------------
    # Stage 1: residual Patch-DVAE
    # ------------------------------------------------------------------
    def _residual_vae_step(
        self,
        batch: dict,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pet = batch["pet"]
        brain = batch["brain"]
        _, _, y_res = decompose_pet(pet)
        target = self.model.extract_image_patches(y_res)
        target_mask = (
            self.model.extract_image_patches(brain)
            if brain is not None
            else torch.ones_like(target)
        )
        reconstructions = []
        mus = []
        logvars = []
        for start in range(0, self.model.num_patches, self.patch_batch_size):
            end = min(start + self.patch_batch_size, self.model.num_patches)
            patch = target[:, start:end]
            batch_size, count = patch.shape[:2]
            reconstruction, mu, logvar, _ = self.model.residual_vae(
                patch.reshape(batch_size * count, patch.shape[2], *patch.shape[-3:])
            )
            reconstructions.append(
                reconstruction.reshape(batch_size, count, *reconstruction.shape[1:])
            )
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
        return loss, {
            "loss": loss.detach(),
            "rec": rec.detach(),
            "lap": lap.detach(),
            "kl": kl.detach(),
        }

    # ------------------------------------------------------------------
    # Stage 2: metabolic prior (MRI -> coarse PET + Parc98 ROI uptake)
    # ------------------------------------------------------------------
    def _metabolic_prior_step(
        self,
        batch: dict,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mri = batch["mri"]
        pet = batch["pet"]
        low_small, _, _ = decompose_pet(pet)
        controls = self.model.mri_control(mri)
        metabolic = self.model.metabolic_prior(controls)
        coarse = metabolic["coarse_low_small"]
        l1 = F.l1_loss(coarse, low_small)
        mse = F.mse_loss(coarse, low_small)
        ssim = 1.0 - self.ssim_loss(coarse, low_small)
        coarse_loss = (
            self._weight(stage_config, "coarse_l1", 1.0) * l1
            + self._weight(stage_config, "coarse_mse", 0.5) * mse
            + self._weight(stage_config, "coarse_ssim", 0.25) * ssim
        )
        gt_region, counts = self._gt_region(pet, batch["parc98"])
        roi, roi_metrics = roi_loss(
            metabolic["region_uptake"],
            gt_region,
            valid=(counts > 0).float(),
            corr_weight=self.roi_corr_weight,
        )
        loss = (
            coarse_loss
            + self._weight(stage_config, "roi", 1.0) * roi
        )
        metrics = {
            "loss": loss.detach(),
            "coarse_l1": l1.detach(),
            "coarse_mse": mse.detach(),
            "coarse_ssim": ssim.detach(),
            "coarse_loss": coarse_loss.detach(),
            "roi": roi.detach(),
        }
        metrics.update({key: value.detach() for key, value in roi_metrics.items()})
        return loss, metrics

    # ------------------------------------------------------------------
    # Stage 3: residual latent diffusion (VAE encoder + metabolic prior frozen)
    # ------------------------------------------------------------------
    def _residual_diffusion_step(
        self,
        batch: dict,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mri = batch["mri"]
        pet = batch["pet"]
        table = batch["table"]
        table_mask = batch["table_mask"]
        _, _, y_res = decompose_pet(pet)
        clean = self.model.encode_residual_patches(
            y_res, self.patch_batch_size, sample=False, encoder_grad=False
        )
        batch_size = clean.shape[0]
        timesteps = torch.randint(
            0, self.schedule.timesteps, (batch_size,), device=self.device
        )
        _, predicted_noise, predicted_clean, _, noise = self.model.forward_hmrd_diffusion(
            clean,
            timesteps,
            mri,
            table,
            table_mask,
            self.patch_batch_size,
            use_table=self.use_table_condition,
            use_global=True,
            use_metabolic_prior=self.use_metabolic_prior,
            use_roi_tokens=self.use_roi_tokens,
            use_anatomical_routing=self.use_anatomical_routing,
        )
        diffusion_loss, clean_loss = self._diffusion_objective(
            timesteps, clean, predicted_noise, predicted_clean, noise
        )
        loss = (
            self._weight(stage_config, "diffusion", 1.0) * diffusion_loss
            + self._weight(stage_config, "clean", 0.1) * clean_loss
        )
        return loss, {
            "loss": loss.detach(),
            "diffusion": diffusion_loss.detach(),
            "clean": clean_loss.detach(),
            "timestep": timesteps.float().mean().detach(),
        }

    # ------------------------------------------------------------------
    # Shared diffusion objective (SNR-weighted noise MSE + predicted-clean)
    # ------------------------------------------------------------------
    def _diffusion_objective(
        self,
        timesteps: torch.Tensor,
        clean: torch.Tensor,
        predicted_noise: torch.Tensor,
        predicted_clean: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.schedule.sqrt_alpha_bars[timesteps].square()
        one_minus = (1.0 - alpha_bar).clamp_min(1e-8)
        snr = alpha_bar / one_minus
        snr_weight = (snr.clamp(max=self.min_snr_gamma) / snr.clamp_min(1e-8)).to(
            predicted_noise.dtype
        )
        noise_error = (predicted_noise - noise).square().flatten(1).mean(1)
        diffusion_loss = (noise_error * snr_weight).mean()
        valid_clean = (
            alpha_bar >= self.clean_alpha_min
        ).float()
        clean_error = F.smooth_l1_loss(
            predicted_clean, clean, reduction="none"
        ).flatten(1).mean(1)
        clean_loss = (clean_error * valid_clean).sum() / valid_clean.sum().clamp_min(1.0)
        return diffusion_loss, clean_loss

    # ------------------------------------------------------------------
    # Stage 4: joint fine-tuning (full HMRD loss)
    # ------------------------------------------------------------------
    def _joint_finetune_step(
        self,
        batch: dict,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mri = batch["mri"]
        pet = batch["pet"]
        table = batch["table"]
        table_mask = batch["table_mask"]
        brain = batch["brain"]
        low_small, _, y_res = decompose_pet(pet)
        clean = self.model.encode_residual_patches(
            y_res, self.patch_batch_size, sample=False, encoder_grad=False
        )
        batch_size = clean.shape[0]
        timesteps = torch.randint(
            0, self.schedule.timesteps, (batch_size,), device=self.device
        )
        _, predicted_noise, predicted_clean, context, noise = self.model.forward_hmrd_diffusion(
            clean,
            timesteps,
            mri,
            table,
            table_mask,
            self.patch_batch_size,
            use_table=self.use_table_condition,
            use_global=True,
            use_metabolic_prior=self.use_metabolic_prior,
            use_roi_tokens=self.use_roi_tokens,
            use_anatomical_routing=self.use_anatomical_routing,
        )
        diffusion_loss, clean_loss = self._diffusion_objective(
            timesteps, clean, predicted_noise, predicted_clean, noise
        )
        # Decode the predicted clean residual latent (decoder is trainable here).
        residual_patches = self.model.decode_residual_latent_patches(
            predicted_clean, self.patch_batch_size
        )
        pred_residual_pet = self.model.stitch_image_patches(residual_patches)
        # Ablation: when use_metabolic_prior=false the context has no coarse PET.
        coarse_pet = context["coarse_pet"]
        if coarse_pet is None:
            coarse_pet = torch.zeros_like(pred_residual_pet)
        pred_pet = (coarse_pet + pred_residual_pet).clamp(-1.0, 1.0)

        if self.use_metabolic_prior:
            coarse_loss = (
                F.l1_loss(context["coarse_low_small"], low_small)
                + 0.5 * F.mse_loss(context["coarse_low_small"], low_small)
                + 0.25 * (1.0 - self.ssim_loss(context["coarse_low_small"], low_small))
            )
            gt_region, counts = self._gt_region(pet, batch["parc98"])
            roi, roi_metrics = roi_loss(
                context["region_uptake"],
                gt_region,
                valid=(counts > 0).float(),
                corr_weight=self.roi_corr_weight,
            )
        else:
            coarse_loss = pred_pet.new_zeros(())
            roi = pred_pet.new_zeros(())
            roi_metrics = {}
        res = F.smooth_l1_loss(pred_residual_pet, y_res)
        image = self._image_reconstruction_loss(pred_pet, pet)
        lap = (
            laplacian_pyramid_loss(
                pred_pet,
                pet,
                levels=self.lap_levels,
                weights=self.lap_weights,
            )
            if self.use_laplacian_loss
            else pred_pet.new_zeros(())
        )
        total = (
            self.lambda_diff * diffusion_loss
            + self.lambda_clean * clean_loss
            + self.lambda_coarse * coarse_loss
            + self.lambda_roi * roi
            + self.lambda_residual * res
            + self.lambda_image * image
            + self.lambda_lap * lap
        )
        metrics = {
            "loss": total.detach(),
            "diffusion": diffusion_loss.detach(),
            "clean": clean_loss.detach(),
            "coarse": coarse_loss.detach(),
            "roi": roi.detach(),
            "residual": res.detach(),
            "image": image.detach(),
            "lap": lap.detach(),
            "timestep": timesteps.float().mean().detach(),
        }
        metrics.update({key: value.detach() for key, value in roi_metrics.items()})
        return total, metrics

    def _image_reconstruction_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """L_image = 1.0*L1 + 0.5*MSE + 0.5*(1-SSIM) + 0.1*MS-SSIM."""
        l1 = F.l1_loss(prediction, target)
        mse = F.mse_loss(prediction, target)
        ssim = 1.0 - self.ssim_loss(prediction, target)
        multiscale = multiscale_ssim_loss(
            prediction, target, self.ssim_loss, factor=2, levels=self.ms_ssim_levels
        )
        return l1 + 0.5 * mse + 0.5 * ssim + 0.1 * multiscale

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _official_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor, brain: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        values = compute_image_metrics_detailed(
            prediction.detach().float().clamp(-1.0, 1.0),
            target.detach().float(),
            mask=brain.detach().float(),
            data_range=self.metric_data_range,
        )
        return {key: prediction.new_tensor(float(value)) for key, value in values.items()}

    def _selection_score(self, metrics: dict[str, float]) -> float:
        weights = self.config["training"].get(
            "selection_weights",
            {"ssim": 0.84, "brain_ssim": 0.10, "rmse": 0.03, "brain_rmse": 0.02, "mae": 0.01},
        )
        if "ssim" not in metrics:
            return float(metrics["loss"])
        return (
            float(weights.get("ssim", 0.84)) * (1.0 - metrics["ssim"])
            + float(weights.get("brain_ssim", 0.10)) * (1.0 - metrics["brain_ssim"])
            + float(weights.get("rmse", 0.03)) * metrics["rmse"]
            + float(weights.get("brain_rmse", 0.02)) * metrics["brain_rmse"]
            + float(weights.get("mae", 0.01)) * metrics["mae"]
        )

    def _hmrd_validation_metrics(
        self, batch: dict, include_official: bool
    ) -> dict[str, torch.Tensor]:
        """Run the official DDIM prediction path and collect every Exp031 metric."""
        mri = batch["mri"]
        pet = batch["pet"]
        table = batch["table"]
        table_mask = batch["table_mask"]
        brain = batch["brain"]
        pred_pet, auxiliary = self.model.predict_hmrd(
            mri,
            table,
            table_mask,
            self.patch_batch_size,
            use_table=self.use_table_condition,
            use_global=True,
            use_metabolic_prior=self.use_metabolic_prior,
            use_roi_tokens=self.use_roi_tokens,
            use_anatomical_routing=self.use_anatomical_routing,
            ddim_steps=self.ddim_steps,
            ddim_eta=self.ddim_eta,
            validation_seed=self.validation_seed,
            clamp_output=True,
        )
        metrics: dict[str, torch.Tensor] = {}
        if include_official:
            metrics.update(self._official_metrics(pred_pet, pet, brain))
        # coarse PET (full-volume, no mask)
        coarse_pet = auxiliary["coarse_pet"]
        metrics["coarse_mae"] = F.l1_loss(coarse_pet, pet)
        metrics["coarse_rmse"] = F.mse_loss(coarse_pet, pet).sqrt()
        metrics["coarse_ssim"] = self.ssim_loss(coarse_pet, pet)
        # residual PET
        _, _, y_res = decompose_pet(pet)
        pred_residual_pet = auxiliary["pred_residual_pet"]
        metrics["residual_mae"] = F.l1_loss(pred_residual_pet, y_res)
        metrics["residual_rmse"] = F.mse_loss(pred_residual_pet, y_res).sqrt()
        # Parc98 ROI
        if auxiliary["region_uptake"] is not None:
            gt_region, counts = self._gt_region(pet, batch["parc98"])
            valid = (counts > 0).float()
            pred_region = auxiliary["region_uptake"]
            metrics["roi_mae"] = F.l1_loss(pred_region * valid, gt_region * valid)
            metrics["roi_rmse"] = F.mse_loss(pred_region * valid, gt_region * valid).sqrt()
            metrics["roi_pearson"] = pearson_correlation(
                pred_region, gt_region, valid=valid
            )
        return metrics

    def _step(
        self, stage: str, batch: dict, stage_config: dict, include_official: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = self._batch(batch)
        if stage == "residual_patch_vae":
            return self._residual_vae_step(values, stage_config)
        if stage == "metabolic_prior":
            return self._metabolic_prior_step(values, stage_config)
        if stage == "residual_diffusion":
            return self._residual_diffusion_step(values, stage_config)
        if stage != "joint_finetune":
            raise ValueError(f"Unknown Exp031 stage: {stage}")
        loss, metrics = self._joint_finetune_step(values, stage_config)
        if include_official:
            with torch.no_grad():
                # Official metrics must be computed in fp32 (same as the
                # independent eval script); fp16/autocast shifts SSIM ~0.005.
                with torch.cuda.amp.autocast(enabled=False):
                    val_metrics = self._hmrd_validation_metrics(values, include_official=True)
                metrics.update(val_metrics)
                metrics["selection_score"] = loss.new_tensor(
                    self._selection_score(
                        {key: float(value.item()) for key, value in val_metrics.items()}
                    )
                )
        return loss, metrics

    # ------------------------------------------------------------------
    # Epoch / run plumbing (mirrors the stable Exp027 trainer)
    # ------------------------------------------------------------------
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
    def _write_best(
        path: Path,
        stage: str,
        epoch: int,
        score: Optional[float],
        metrics: Optional[dict],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {"stage": stage, "epoch": int(epoch), "score": score, "metrics": metrics},
                handle,
                indent=2,
            )
        temporary.replace(path)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask).sum() / mask.sum().clamp_min(1e-6)

    @staticmethod
    def _laplacian_pyramid_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Two-scale 3D Laplacian pyramid loss on residual patches (stage 1)."""
        current_prediction = prediction.reshape(
            -1, prediction.shape[2], *prediction.shape[-3:]
        )
        current_target = target.reshape(-1, target.shape[2], *target.shape[-3:])
        result = current_prediction.new_zeros(())
        for level in range(2):
            down_prediction = F.avg_pool3d(current_prediction, 2, 2)
            down_target = F.avg_pool3d(current_target, 2, 2)
            up_prediction = F.interpolate(
                down_prediction,
                size=current_prediction.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            up_target = F.interpolate(
                down_target,
                size=current_target.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            result = result + (2.0 ** (-level)) * F.l1_loss(
                current_prediction - up_prediction,
                current_target - up_target,
            )
            current_prediction, current_target = down_prediction, down_target
        return result + 0.25 * F.l1_loss(current_prediction, current_target)

    @contextmanager
    def _validation_rng(self):
        devices = []
        if self.device.type == "cuda":
            devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(self.validation_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.validation_seed)
            yield

    def _run_epoch(
        self,
        stage: str,
        stage_config: dict,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        scaler: torch.cuda.amp.GradScaler,
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        set_stage_mode(self.model, stage, training=training)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(
            loader,
            desc=f"exp031 {stage} {epoch} {'train' if training else 'val'}",
            leave=False,
        )
        context = nullcontext() if training else self._validation_rng()
        with context:
            for batch in progress:
                batch_size = int(batch["image"].shape[0])
                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        loss, metrics = self._step(
                            stage, batch, stage_config, include_official=False
                        )
                    scaler.scale(loss).backward()
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.gradient_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.amp):
                            _, metrics = self._step(
                                stage, batch, stage_config, include_official=True
                            )
                seen += batch_size
                for name, value in metrics.items():
                    totals[name] += float(value.detach().item()) * batch_size
                progress.set_postfix(loss=float(metrics["loss"].detach().item()))
        if seen == 0:
            raise RuntimeError(f"No valid batches were produced for Exp031 stage {stage}")
        return {name: value / seen for name, value in totals.items()}

    def run(self, only_stage: Optional[str] = None) -> None:
        if only_stage is not None and only_stage not in STAGE_NAMES:
            raise ValueError(f"Unknown Exp031 stage: {only_stage}")
        if only_stage is not None:
            stage_index = STAGE_NAMES.index(only_stage)
            if stage_index > 0:
                for previous in STAGE_NAMES[:stage_index]:
                    previous_config = self.config["stages"].get(previous, {})
                    if not previous_config.get("enabled", True) or int(
                        previous_config.get("epochs", 0)
                    ) == 0:
                        continue
                    checkpoint = (
                        self.output_root / previous / "checkpoints" / "best.pt"
                    )
                    if not checkpoint.exists():
                        raise RuntimeError(
                            f"Exp031 stage {only_stage} requires previous best checkpoint: {checkpoint}"
                        )
        for stage in STAGE_NAMES:
            if only_stage is not None and stage != only_stage:
                continue
            stage_config = self.config["stages"][stage]
            epochs = int(stage_config.get("epochs", 0))
            if not stage_config.get("enabled", True) or epochs == 0:
                print(
                    f"[exp031] skip {stage}: enabled={stage_config.get('enabled', True)}, epochs={epochs}"
                )
                continue
            groups = configure_trainable_modules(self.model, stage)
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            if not parameters:
                raise RuntimeError(f"Exp031 stage {stage} has no trainable parameters")
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
            best_score = float("inf")
            if resume:
                checkpoint = load_checkpoint(
                    resume, self.model, optimizer=optimizer, map_location=self.device, strict=True
                )
                if checkpoint.get("stage") != stage:
                    raise ValueError(
                        f"Resume checkpoint stage={checkpoint.get('stage')} does not match {stage}"
                    )
                start_epoch = int(checkpoint["epoch"]) + 1
                best_score = self._read_best(stage_dir / "best_metric.json")
            best_ssim = -float("inf")
            best_psnr = -float("inf")
            best_brain_ssim = -float("inf")
            print(
                f"[exp031] stage={stage}, epochs={epochs}, trainable_groups={groups}, "
                f"trainable_params={sum(p.numel() for p in parameters)}"
            )
            for epoch in range(start_epoch, epochs + 1):
                train_metrics = self._run_epoch(
                    stage, stage_config, self.train_loader, optimizer, scaler, epoch, True
                )
                val_metrics = None
                interval = max(1, int(self.config["training"].get("val_interval", 1)))
                if self.val_loader is not None and (
                    epoch % interval == 0 or epoch == epochs
                ):
                    val_metrics = self._run_epoch(
                        stage, stage_config, self.val_loader, None, scaler, epoch, False
                    )
                record = {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_metrics,
                    "val": val_metrics,
                }
                self._append_metrics(stage_dir / "metrics.jsonl", record)
                monitor = val_metrics if val_metrics is not None else train_metrics
                score = self._selection_score(monitor)
                scheduler.step(score)
                if score < best_score:
                    best_score = score
                    save_checkpoint(
                        checkpoints / "best.pt",
                        self.model,
                        optimizer,
                        stage,
                        epoch,
                        self.config,
                        record,
                    )
                    self._write_best(stage_dir / "best_metric.json", stage, epoch, score, monitor)
                if val_metrics is not None:
                    if val_metrics.get("ssim", -float("inf")) > best_ssim:
                        best_ssim = val_metrics["ssim"]
                        save_checkpoint(
                            checkpoints / "best_ssim.pt",
                            self.model,
                            optimizer,
                            stage,
                            epoch,
                            self.config,
                            record,
                        )
                    if val_metrics.get("psnr", -float("inf")) > best_psnr:
                        best_psnr = val_metrics["psnr"]
                        save_checkpoint(
                            checkpoints / "best_psnr.pt",
                            self.model,
                            optimizer,
                            stage,
                            epoch,
                            self.config,
                            record,
                        )
                    if val_metrics.get("brain_ssim", -float("inf")) > best_brain_ssim:
                        best_brain_ssim = val_metrics["brain_ssim"]
                        save_checkpoint(
                            checkpoints / "best_brain_ssim.pt",
                            self.model,
                            optimizer,
                            stage,
                            epoch,
                            self.config,
                            record,
                        )
                save_checkpoint(
                    checkpoints / "last.pt",
                    self.model,
                    optimizer,
                    stage,
                    epoch,
                    self.config,
                    record,
                )
                print(f"[exp031] {stage} epoch={epoch} train={train_metrics} val={val_metrics}")
            best_path = checkpoints / "best.pt"
            if not best_path.exists():
                raise FileNotFoundError(f"Exp031 stage {stage} finished without {best_path}")
            selected = load_checkpoint(best_path, self.model, map_location=self.device, strict=True)
            print(f"[exp031] stage={stage} selected best.pt from epoch={selected['epoch']}")

__all__ = ["Exp031Trainer"]
