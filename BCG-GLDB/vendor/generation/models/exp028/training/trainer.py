"""Exp028 trainer extensions over the audited Exp027 trainer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from exp027.training.trainer import Exp027Trainer
from exp028.losses.pet_fidelity import (
    low_frequency_l1_loss,
    mri_edge_guided_gradient_loss,
)


class Exp028Trainer(Exp027Trainer):
    """SSIM-selected Exp027 with bounded MRI-guided PET fidelity terms."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_mri: Optional[torch.Tensor] = None

    def _patch_vae_step_with_reconstruction(
        self,
        pet: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
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
            reconstructions.append(
                reconstruction.reshape(batch_size, count, *reconstruction.shape[1:])
            )
            mus.append(mu)
            logvars.append(logvar)
        reconstruction_patches = torch.cat(reconstructions, dim=1)
        mu = torch.cat(mus, dim=0)
        logvar = torch.cat(logvars, dim=0)
        rec = self._masked_mean(
            (reconstruction_patches - target).abs(), target_mask
        )
        lap = self._laplacian_pyramid_loss(reconstruction_patches, target)
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
        loss = (
            self._weight(stage_config, "rec", 1.0) * rec
            + self._weight(stage_config, "lap", 0.1) * lap
            + self._weight(stage_config, "kl", 1e-6) * kl
        )
        reconstruction = self.model.stitch_image_patches(reconstruction_patches)
        metrics = {
            "loss": loss.detach(),
            "rec": rec.detach(),
            "lap": lap.detach(),
            "kl": kl.detach(),
        }
        return loss, metrics, reconstruction

    def _official_proxy_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        brain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._official_metrics(prediction, target, brain)

    def _mri_prior_proxy(
        self, mri: torch.Tensor, pet: torch.Tensor, brain: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        controls = self.model.mri_control(mri)
        levels = self.model.mri_prior_heads(
            self.model.build_control_patch_pyramid(controls)
        )
        decoded = self.model.decode_latent_patches(
            levels["high"], self.patch_batch_size
        )
        prediction = self.model.stitch_image_patches(decoded)
        return self._official_proxy_metrics(prediction, pet, brain)

    def _local_diffusion_proxy(
        self, mri: torch.Tensor, pet: torch.Tensor, brain: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        clean = self._target_latent(pet)
        timesteps = torch.zeros(clean.shape[0], device=self.device, dtype=torch.long)
        predicted_noise, _ = self.model.denoise_patches(
            clean,
            timesteps,
            mri,
            None,
            None,
            self.patch_batch_size,
            use_table=False,
            use_global=False,
            inject_table_locally=False,
        )
        predicted_clean = self.schedule.predict_clean(
            clean, timesteps, predicted_noise
        )
        decoded = self.model.decode_latent_patches(
            predicted_clean, self.patch_batch_size
        )
        prediction = self.model.stitch_image_patches(decoded)
        return self._official_proxy_metrics(prediction, pet, brain)

    def _image_terms(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, values = super()._image_terms(
            prediction, target, brain_mask, stage_config
        )
        edge_weight = self._weight(stage_config, "mri_edge", 0.0)
        low_frequency_weight = self._weight(stage_config, "low_frequency", 0.0)
        if edge_weight == 0.0 and low_frequency_weight == 0.0:
            return total, values
        if self._current_mri is None:
            raise RuntimeError("Exp028 MRI-guided loss requires the current MRI batch")
        mri = self._current_mri.to(device=prediction.device, dtype=prediction.dtype)
        if edge_weight != 0.0:
            edge = mri_edge_guided_gradient_loss(
                prediction,
                target,
                mri,
                brain_mask,
                edge_strength=float(
                    self.config["training"].get("mri_edge_strength", 0.35)
                ),
            )
            total = total + edge_weight * edge
            values["mri_edge"] = edge.detach()
        if low_frequency_weight != 0.0:
            low_frequency = low_frequency_l1_loss(
                prediction,
                target,
                factor=int(self.config["training"].get("low_frequency_factor", 2)),
            )
            total = total + low_frequency_weight * low_frequency
            values["low_frequency"] = low_frequency.detach()
        values["loss_image"] = total.detach()
        return total, values

    def _selection_score(self, metrics: dict[str, float]) -> float:
        # Stage 1 is explicitly selected by reconstructed-PET SSIM.  The
        # remaining stages use the configured SSIM-first composite score.
        if "stage1_ssim" in metrics:
            return 1.0 - float(metrics["stage1_ssim"])
        return super()._selection_score(metrics)

    def _step(
        self,
        stage: str,
        batch: dict,
        stage_config: dict,
        include_official: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if stage == "patch_vae":
            _, pet, _, _ = self._batch(batch)
            brain = self._brain_mask(batch, pet)
            loss, metrics, reconstruction = self._patch_vae_step_with_reconstruction(
                pet, brain, stage_config
            )
            if include_official:
                official = self._official_proxy_metrics(reconstruction, pet, brain)
                metrics.update(official)
                metrics["stage1_ssim"] = official["ssim"]
            return loss, metrics

        if stage == "mri_patch_prior":
            loss, metrics = super()._step(
                stage, batch, stage_config, include_official=False
            )
            if include_official:
                mri, pet, _, _ = self._batch(batch)
                brain = self._brain_mask(batch, pet)
                metrics.update(self._mri_prior_proxy(mri, pet, brain))
            return loss, metrics

        if stage == "local_diffusion":
            loss, metrics = super()._step(
                stage, batch, stage_config, include_official=False
            )
            if include_official:
                mri, pet, _, _ = self._batch(batch)
                brain = self._brain_mask(batch, pet)
                metrics.update(self._local_diffusion_proxy(mri, pet, brain))
            return loss, metrics

        self._current_mri = batch["image"].to(self.device, non_blocking=True).float()
        try:
            return super()._step(stage, batch, stage_config, include_official)
        finally:
            self._current_mri = None


__all__ = ["Exp028Trainer"]
