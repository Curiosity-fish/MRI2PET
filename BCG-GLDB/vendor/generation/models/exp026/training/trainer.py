from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp026.models.exp026_model import Exp026MRI2PET
from exp026.training.diffusion import DiffusionSchedule
from exp026.training.stages import (
    configure_trainable_modules,
    set_stage_mode,
    trainable_parameter_names,
)
from exp026.utils.checkpoints import load_checkpoint, save_checkpoint
from exp026.utils.config import STAGE_NAMES
from exp026.utils.losses import gradient_matching_loss, kl_divergence, patch_seam_loss


class Exp026Trainer:
    def __init__(
        self,
        model: Exp026MRI2PET,
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

    def _batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mri = batch["image"].to(self.device, non_blocking=True).float()
        pet = batch["label"].to(self.device, non_blocking=True).float()
        table = batch["table"].to(self.device, non_blocking=True).float()
        mask = batch.get("mask")
        if mask is None:
            mask = torch.zeros_like(table)
        else:
            mask = mask.to(self.device, non_blocking=True).float()
        return mri, pet, table, mask

    @staticmethod
    def _weight(stage_config: dict, name: str, default: float = 0.0) -> float:
        return float(stage_config.get("loss", {}).get(name, default))

    def _capture_rng_state(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cuda_state = None
        if self.device.type == "cuda":
            cuda_state = torch.cuda.get_rng_state(self.device)
        return torch.get_rng_state(), cuda_state

    def _restore_rng_state(
        self,
        state: tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> None:
        cpu_state, cuda_state = state
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, self.device)

    def _mri_prior_consistency(
        self,
        control_patches: dict[str, torch.Tensor],
        clean: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        targets = self.model.build_latent_patch_pyramid(clean)
        predictions = self.model.mri_prior_heads(control_patches)
        losses = {
            level: F.l1_loss(predictions[level], targets[level])
            for level in ("high", "middle", "low")
        }
        combined = losses["high"] + 0.5 * losses["middle"] + 0.25 * losses["low"]
        return combined, losses

    def _patch_vae_step(self, pet: torch.Tensor, stage_config: dict) -> tuple[torch.Tensor, dict]:
        target = self.model.extract_image_patches(pet)
        reconstructions = []
        mus = []
        logvars = []
        for start in range(0, self.model.num_patches, self.patch_batch_size):
            patch = target[:, start : start + self.patch_batch_size]
            batch, count = patch.shape[:2]
            flat = patch.reshape(batch * count, patch.shape[2], *patch.shape[-3:])
            reconstruction, mu, logvar, _ = self.model.patch_vae(flat)
            reconstructions.append(reconstruction.reshape(batch, count, *reconstruction.shape[1:]))
            mus.append(mu)
            logvars.append(logvar)
        reconstruction_patches = torch.cat(reconstructions, dim=1)
        patch_loss = F.l1_loss(reconstruction_patches, target)
        gradient_loss = gradient_matching_loss(reconstruction_patches, target)
        kl_loss = kl_divergence(torch.cat(mus, dim=0), torch.cat(logvars, dim=0))
        loss = (
            self._weight(stage_config, "patch_reconstruction", 1.0) * patch_loss
            + self._weight(stage_config, "gradient", 0.0) * gradient_loss
            + self._weight(stage_config, "kl", 1e-6) * kl_loss
        )
        return loss, {
            "loss": loss.detach(),
            "patch_reconstruction": patch_loss.detach(),
            "gradient": gradient_loss.detach(),
            "kl": kl_loss.detach(),
        }

    def _patch_vae_train_step(
        self,
        pet: torch.Tensor,
        stage_config: dict,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict[str, torch.Tensor]:
        target = self.model.extract_image_patches(pet)
        metrics = {
            "patch_reconstruction": pet.new_zeros(()),
            "gradient": pet.new_zeros(()),
            "kl": pet.new_zeros(()),
        }
        for start in range(0, self.model.num_patches, self.patch_batch_size):
            end = min(start + self.patch_batch_size, self.model.num_patches)
            patch = target[:, start:end]
            batch, count = patch.shape[:2]
            fraction = float(count) / float(self.model.num_patches)
            flat = patch.reshape(batch * count, patch.shape[2], *patch.shape[-3:])
            with torch.cuda.amp.autocast(enabled=self.amp):
                reconstruction, mu, logvar, _ = self.model.patch_vae(flat)
                reconstruction = reconstruction.reshape(
                    batch, count, *reconstruction.shape[1:]
                )
                patch_loss = F.l1_loss(reconstruction, patch)
                gradient_loss = gradient_matching_loss(reconstruction, patch)
                kl_loss = kl_divergence(mu, logvar)
                chunk_loss = fraction * (
                    self._weight(stage_config, "patch_reconstruction", 1.0) * patch_loss
                    + self._weight(stage_config, "gradient", 0.0) * gradient_loss
                    + self._weight(stage_config, "kl", 1e-6) * kl_loss
                )
            scaler.scale(chunk_loss).backward()
            metrics["patch_reconstruction"] += fraction * patch_loss.detach()
            metrics["gradient"] += fraction * gradient_loss.detach()
            metrics["kl"] += fraction * kl_loss.detach()
        metrics["loss"] = (
            self._weight(stage_config, "patch_reconstruction", 1.0)
            * metrics["patch_reconstruction"]
            + self._weight(stage_config, "gradient", 0.0) * metrics["gradient"]
            + self._weight(stage_config, "kl", 1e-6) * metrics["kl"]
        )
        return metrics

    def _mri_patch_prior_step(
        self,
        mri: torch.Tensor,
        pet: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict]:
        target_high = self.model.encode_pet_patches(
            pet, self.patch_batch_size, sample=False, encoder_grad=False
        )
        targets = self.model.build_latent_patch_pyramid(target_high)
        controls = self.model.mri_control(mri)
        control_patches = self.model.build_control_patch_pyramid(controls)
        predictions = self.model.mri_prior_heads(control_patches)

        high_loss = F.l1_loss(predictions["high"], targets["high"])
        middle_loss = F.l1_loss(predictions["middle"], targets["middle"])
        low_loss = F.l1_loss(predictions["low"], targets["low"])
        gradient_loss = torch.stack(
            [
                gradient_matching_loss(predictions[level], targets[level])
                for level in ("high", "middle", "low")
            ]
        ).mean()
        loss = (
            self._weight(stage_config, "high", 1.0) * high_loss
            + self._weight(stage_config, "middle", 0.5) * middle_loss
            + self._weight(stage_config, "low", 0.25) * low_loss
            + self._weight(stage_config, "gradient", 0.0) * gradient_loss
        )
        return loss, {
            "loss": loss.detach(),
            "high": high_loss.detach(),
            "middle": middle_loss.detach(),
            "low": low_loss.detach(),
            "gradient": gradient_loss.detach(),
        }

    def _table_prior_step(
        self,
        mri: torch.Tensor,
        pet: torch.Tensor,
        table: torch.Tensor,
        mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict]:
        target_patches = self.model.encode_pet_patches(
            pet, self.patch_batch_size, sample=False, encoder_grad=False
        )
        with torch.no_grad():
            controls = self.model.mri_control(mri)
            control_patches = self.model.build_control_patch_pyramid(controls)
            base_prior = self.model.mri_prior_heads(control_patches)["high"]
        context = self.model.table_interaction(
            control_patches["high"],
            table,
            mask,
            base_prior_patches=base_prior,
        )
        prior_loss = F.l1_loss(context["prior_patches"], target_patches)
        gradient_loss = gradient_matching_loss(context["prior_patches"], target_patches)
        loss = (
            self._weight(stage_config, "prior", 1.0) * prior_loss
            + self._weight(stage_config, "gradient", 0.0) * gradient_loss
        )
        return loss, {
            "loss": loss.detach(),
            "prior": prior_loss.detach(),
            "gradient": gradient_loss.detach(),
            "table_reliability": context["reliability"].mean().detach(),
        }

    def _diffusion_step(
        self,
        stage: str,
        mri: torch.Tensor,
        pet: torch.Tensor,
        table: torch.Tensor,
        mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict]:
        clean = self.model.encode_pet_patches(
            pet, self.patch_batch_size, sample=False, encoder_grad=False
        )
        batch = clean.shape[0]
        timesteps = torch.randint(
            0, self.schedule.timesteps, (batch,), device=self.device, dtype=torch.long
        )
        noise = torch.randn_like(clean)
        noisy = self.schedule.q_sample(clean, timesteps, noise)
        use_table = stage in ("global_mamba", "joint_finetune")
        use_global = stage in ("global_mamba", "joint_finetune")
        inject_table = stage == "joint_finetune"
        predicted_noise, auxiliary = self.model.denoise_patches(
            noisy,
            timesteps,
            mri,
            table,
            mask,
            self.patch_batch_size,
            use_table=use_table,
            use_global=use_global,
            inject_table_locally=inject_table,
        )
        predicted_clean = self.schedule.predict_clean(noisy, timesteps, predicted_noise)
        noise_loss = F.mse_loss(predicted_noise, noise)
        latent_loss = F.l1_loss(predicted_clean, clean)
        loss = (
            self._weight(stage_config, "noise", 1.0) * noise_loss
            + self._weight(stage_config, "latent", 0.0) * latent_loss
        )
        metrics = {
            "noise": noise_loss.detach(),
            "latent": latent_loss.detach(),
        }
        if stage == "local_control":
            mri_prior_loss, prior_levels = self._mri_prior_consistency(
                auxiliary["control_patches"], clean
            )
            loss = loss + self._weight(stage_config, "mri_prior", 0.0) * mri_prior_loss
            metrics.update(
                {
                    "mri_prior": mri_prior_loss.detach(),
                    "mri_prior_high": prior_levels["high"].detach(),
                    "mri_prior_middle": prior_levels["middle"].detach(),
                    "mri_prior_low": prior_levels["low"].detach(),
                }
            )
        if stage == "joint_finetune":
            context = auxiliary["table_context"]
            prior_loss = F.l1_loss(context["prior_patches"], clean)
            decoded_patches = self.model.decode_latent_patches(
                predicted_clean, self.patch_batch_size
            )
            predicted_pet = self.model.stitch_image_patches(decoded_patches)
            full_pet_loss = F.l1_loss(predicted_pet, pet)
            seam_loss = patch_seam_loss(predicted_clean, clean, self.model.grid_size)
            loss = (
                loss
                + self._weight(stage_config, "prior", 0.0) * prior_loss
                + self._weight(stage_config, "full_pet", 0.0) * full_pet_loss
                + self._weight(stage_config, "seam", 0.0) * seam_loss
            )
            metrics.update(
                {
                    "prior": prior_loss.detach(),
                    "full_pet": full_pet_loss.detach(),
                    "seam": seam_loss.detach(),
                }
            )
        metrics["loss"] = loss.detach()
        return loss, metrics

    def _streaming_diffusion_train_step(
        self,
        stage: str,
        mri: torch.Tensor,
        pet: torch.Tensor,
        table: torch.Tensor,
        mask: torch.Tensor,
        stage_config: dict,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.amp):
            clean = self.model.encode_pet_patches(
                pet, self.patch_batch_size, sample=False, encoder_grad=False
            )
        batch = clean.shape[0]
        timesteps = torch.randint(
            0, self.schedule.timesteps, (batch,), device=self.device, dtype=torch.long
        )
        noise = torch.randn_like(clean)
        noisy = self.schedule.q_sample(clean, timesteps, noise)
        use_table = stage in ("global_mamba", "joint_finetune")
        use_global = stage in ("global_mamba", "joint_finetune")
        inject_table = stage == "joint_finetune"
        with torch.cuda.amp.autocast(enabled=self.amp):
            denoise_context = self.model.prepare_denoise_context(
                noisy,
                timesteps,
                mri,
                table,
                mask,
                use_table=use_table,
                use_global=use_global,
                inject_table_locally=inject_table,
            )

        mri_prior_loss = None
        mri_prior_levels = None
        mri_prior_metric = None
        mri_prior_level_metrics = None
        if stage == "local_control":
            with torch.cuda.amp.autocast(enabled=self.amp):
                mri_prior_loss, mri_prior_levels = self._mri_prior_consistency(
                    denoise_context["control_patches"], clean
                )
            mri_prior_metric = mri_prior_loss.detach()
            mri_prior_level_metrics = {
                level: value.detach() for level, value in mri_prior_levels.items()
            }

        seam_weight = self._weight(stage_config, "seam", 0.0)
        seam_value = clean.new_zeros((), dtype=torch.float32)
        seam_gradient = None
        replay_states: list[tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        final_rng_state = None
        if stage == "joint_finetune" and seam_weight != 0.0:
            detached_clean_chunks = []
            for start in range(0, self.model.num_patches, self.patch_batch_size):
                end = min(start + self.patch_batch_size, self.model.num_patches)
                replay_states.append(self._capture_rng_state())
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        predicted_noise = self.model.denoise_patch_chunk(
                            noisy, timesteps, denoise_context, start, end
                        )
                        predicted_clean = self.schedule.predict_clean(
                            noisy[:, start:end], timesteps, predicted_noise
                        )
                detached_clean_chunks.append(predicted_clean.detach())
            final_rng_state = self._capture_rng_state()
            seam_probe = torch.cat(detached_clean_chunks, dim=1).float().requires_grad_(True)
            seam_value = patch_seam_loss(
                seam_probe,
                clean.detach().float(),
                self.model.grid_size,
            )
            seam_gradient = torch.autograd.grad(seam_value, seam_probe)[0].to(clean.dtype)
            seam_value = seam_value.detach()
            del seam_probe, detached_clean_chunks

        metrics = {
            "noise": clean.new_zeros(()),
            "latent": clean.new_zeros(()),
        }
        if stage == "joint_finetune":
            metrics.update(
                {
                    "prior": clean.new_zeros(()),
                    "full_pet": clean.new_zeros(()),
                    "seam": seam_value.detach(),
                }
            )
            target_image_patches = self.model.extract_image_patches(pet)

        for chunk_index, start in enumerate(
            range(0, self.model.num_patches, self.patch_batch_size)
        ):
            end = min(start + self.patch_batch_size, self.model.num_patches)
            count = end - start
            fraction = float(count) / float(self.model.num_patches)
            if replay_states:
                self._restore_rng_state(replay_states[chunk_index])
            with torch.cuda.amp.autocast(enabled=self.amp):
                predicted_noise = self.model.denoise_patch_chunk(
                    noisy, timesteps, denoise_context, start, end
                )
                noisy_chunk = noisy[:, start:end]
                clean_chunk = clean[:, start:end]
                noise_chunk = noise[:, start:end]
                predicted_clean = self.schedule.predict_clean(
                    noisy_chunk, timesteps, predicted_noise
                )
                noise_loss = F.mse_loss(predicted_noise, noise_chunk)
                latent_loss = F.l1_loss(predicted_clean, clean_chunk)
                chunk_loss = fraction * (
                    self._weight(stage_config, "noise", 1.0) * noise_loss
                    + self._weight(stage_config, "latent", 0.0) * latent_loss
                )

                if stage == "local_control" and start == 0 and mri_prior_loss is not None:
                    chunk_loss = chunk_loss + self._weight(
                        stage_config, "mri_prior", 0.0
                    ) * mri_prior_loss

                if stage == "joint_finetune":
                    table_context = denoise_context["table_context"]
                    prior_loss = F.l1_loss(
                        table_context["prior_patches"][:, start:end], clean_chunk
                    )
                    decoded_chunk = self.model.decode_latent_chunk(predicted_clean)
                    full_pet_loss = F.l1_loss(
                        decoded_chunk, target_image_patches[:, start:end]
                    )
                    chunk_loss = chunk_loss + fraction * (
                        self._weight(stage_config, "prior", 0.0) * prior_loss
                        + self._weight(stage_config, "full_pet", 0.0) * full_pet_loss
                    )
                    if seam_gradient is not None:
                        seam_surrogate = (
                            predicted_clean.float()
                            * seam_gradient[:, start:end].float()
                        ).sum()
                        chunk_loss = chunk_loss + seam_weight * seam_surrogate

            scaler.scale(chunk_loss).backward(
                retain_graph=end < self.model.num_patches
            )
            if stage == "local_control" and start == 0:
                mri_prior_loss = None
                mri_prior_levels = None
            metrics["noise"] += fraction * noise_loss.detach()
            metrics["latent"] += fraction * latent_loss.detach()
            if stage == "joint_finetune":
                metrics["prior"] += fraction * prior_loss.detach()
                metrics["full_pet"] += fraction * full_pet_loss.detach()
                del prior_loss, decoded_chunk, full_pet_loss
                if seam_gradient is not None:
                    del seam_surrogate
            del chunk_loss, predicted_noise, predicted_clean, noise_loss, latent_loss

        if final_rng_state is not None:
            self._restore_rng_state(final_rng_state)

        if stage == "local_control" and mri_prior_metric is not None:
            metrics.update(
                {
                    "mri_prior": mri_prior_metric,
                    "mri_prior_high": mri_prior_level_metrics["high"],
                    "mri_prior_middle": mri_prior_level_metrics["middle"],
                    "mri_prior_low": mri_prior_level_metrics["low"],
                }
            )

        total = (
            self._weight(stage_config, "noise", 1.0) * metrics["noise"]
            + self._weight(stage_config, "latent", 0.0) * metrics["latent"]
        )
        if stage == "local_control":
            total = total + self._weight(stage_config, "mri_prior", 0.0) * metrics[
                "mri_prior"
            ]
        if stage == "joint_finetune":
            total = (
                total
                + self._weight(stage_config, "prior", 0.0) * metrics["prior"]
                + self._weight(stage_config, "full_pet", 0.0) * metrics["full_pet"]
                + seam_weight * metrics["seam"]
            )
        metrics["loss"] = total.detach()
        return metrics

    def _decoder_step(
        self,
        mri: torch.Tensor,
        pet: torch.Tensor,
        table: torch.Tensor,
        mask: torch.Tensor,
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            clean = self.model.encode_pet_patches(
                pet, self.patch_batch_size, sample=False, encoder_grad=False
            )
            batch = clean.shape[0]
            timesteps = torch.randint(
                0, self.schedule.timesteps, (batch,), device=self.device, dtype=torch.long
            )
            noise = torch.randn_like(clean)
            noisy = self.schedule.q_sample(clean, timesteps, noise)
            predicted_noise, _ = self.model.denoise_patches(
                noisy,
                timesteps,
                mri,
                table,
                mask,
                self.patch_batch_size,
                use_table=True,
                use_global=True,
                inject_table_locally=True,
            )
            predicted_clean = self.schedule.predict_clean(noisy, timesteps, predicted_noise)
        target_patches = self.model.extract_image_patches(pet)
        real_decoded = self.model.decode_latent_patches(clean.detach(), self.patch_batch_size)
        predicted_decoded = self.model.decode_latent_patches(
            predicted_clean.detach(), self.patch_batch_size
        )
        real_loss = F.l1_loss(real_decoded, target_patches)
        predicted_loss = F.l1_loss(predicted_decoded, target_patches)
        loss = (
            self._weight(stage_config, "real_latent", 1.0) * real_loss
            + self._weight(stage_config, "predicted_latent", 0.1) * predicted_loss
        )
        return loss, {
            "loss": loss.detach(),
            "real_latent": real_loss.detach(),
            "predicted_latent": predicted_loss.detach(),
        }

    def _decoder_train_step(
        self,
        mri: torch.Tensor,
        pet: torch.Tensor,
        table: torch.Tensor,
        mask: torch.Tensor,
        stage_config: dict,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp):
                clean = self.model.encode_pet_patches(
                    pet, self.patch_batch_size, sample=False, encoder_grad=False
                )
                batch = clean.shape[0]
                timesteps = torch.randint(
                    0,
                    self.schedule.timesteps,
                    (batch,),
                    device=self.device,
                    dtype=torch.long,
                )
                noise = torch.randn_like(clean)
                noisy = self.schedule.q_sample(clean, timesteps, noise)
                predicted_noise, _ = self.model.denoise_patches(
                    noisy,
                    timesteps,
                    mri,
                    table,
                    mask,
                    self.patch_batch_size,
                    use_table=True,
                    use_global=True,
                    inject_table_locally=True,
                )
                predicted_clean = self.schedule.predict_clean(
                    noisy, timesteps, predicted_noise
                )

        target_patches = self.model.extract_image_patches(pet)
        metrics = {
            "real_latent": pet.new_zeros(()),
            "predicted_latent": pet.new_zeros(()),
        }
        for start in range(0, self.model.num_patches, self.patch_batch_size):
            end = min(start + self.patch_batch_size, self.model.num_patches)
            count = end - start
            fraction = float(count) / float(self.model.num_patches)
            with torch.cuda.amp.autocast(enabled=self.amp):
                real_decoded = self.model.decode_latent_chunk(clean[:, start:end].detach())
                predicted_decoded = self.model.decode_latent_chunk(
                    predicted_clean[:, start:end].detach()
                )
                real_loss = F.l1_loss(real_decoded, target_patches[:, start:end])
                predicted_loss = F.l1_loss(
                    predicted_decoded, target_patches[:, start:end]
                )
                chunk_loss = fraction * (
                    self._weight(stage_config, "real_latent", 1.0) * real_loss
                    + self._weight(stage_config, "predicted_latent", 0.1)
                    * predicted_loss
                )
            scaler.scale(chunk_loss).backward()
            metrics["real_latent"] += fraction * real_loss.detach()
            metrics["predicted_latent"] += fraction * predicted_loss.detach()
        metrics["loss"] = (
            self._weight(stage_config, "real_latent", 1.0) * metrics["real_latent"]
            + self._weight(stage_config, "predicted_latent", 0.1)
            * metrics["predicted_latent"]
        )
        return metrics

    def _train_step(
        self,
        stage: str,
        batch: dict,
        stage_config: dict,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict[str, torch.Tensor]:
        mri, pet, table, mask = self._batch(batch)
        if stage == "patch_vae":
            return self._patch_vae_train_step(pet, stage_config, scaler)
        if stage in ("local_control", "global_mamba", "joint_finetune"):
            return self._streaming_diffusion_train_step(
                stage, mri, pet, table, mask, stage_config, scaler
            )
        if stage == "decoder_calibration":
            return self._decoder_train_step(
                mri, pet, table, mask, stage_config, scaler
            )
        with torch.cuda.amp.autocast(enabled=self.amp):
            if stage == "mri_patch_prior":
                loss, metrics = self._mri_patch_prior_step(mri, pet, stage_config)
            elif stage == "table_mri_prior":
                loss, metrics = self._table_prior_step(
                    mri, pet, table, mask, stage_config
                )
            else:
                raise ValueError(f"Unknown training stage: {stage}")
        scaler.scale(loss).backward()
        return metrics

    def _step(self, stage: str, batch: dict, stage_config: dict) -> tuple[torch.Tensor, dict]:
        mri, pet, table, mask = self._batch(batch)
        if stage == "patch_vae":
            return self._patch_vae_step(pet, stage_config)
        if stage == "mri_patch_prior":
            return self._mri_patch_prior_step(mri, pet, stage_config)
        if stage == "table_mri_prior":
            return self._table_prior_step(mri, pet, table, mask, stage_config)
        if stage == "decoder_calibration":
            return self._decoder_step(mri, pet, table, mask, stage_config)
        return self._diffusion_step(stage, mri, pet, table, mask, stage_config)

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
        set_stage_mode(self.model, stage, training)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(loader, desc=f"exp026 {stage} {epoch} {'train' if training else 'val'}", leave=False)
        rng_context = nullcontext() if training else self._validation_rng()
        with rng_context:
            for batch in progress:
                if batch is None:
                    continue
                batch_size = int(batch["image"].shape[0])
                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    metrics = self._train_step(stage, batch, stage_config, scaler)
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.amp):
                            _, metrics = self._step(stage, batch, stage_config)
                if training and optimizer is not None:
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.gradient_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                seen += batch_size
                for name, value in metrics.items():
                    totals[name] += float(value.item()) * batch_size
                progress.set_postfix(loss=float(metrics["loss"].item()))
        if seen == 0:
            raise RuntimeError(f"No valid batches were produced for stage {stage}")
        return {name: value / seen for name, value in totals.items()}

    def _append_metrics(self, stage: str, record: dict) -> None:
        stage_dir = self.output_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        with (stage_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    @contextmanager
    def _validation_rng(self):
        devices = []
        if self.device.type == "cuda":
            device_index = self.device.index
            devices = [device_index if device_index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = int(self.config["training"].get("validation_seed", 2601))
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            yield

    @staticmethod
    def _read_best_metric(path: Path) -> float:
        if not path.exists():
            return float("inf")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        score = payload.get("score")
        return float(score) if score is not None else float("inf")

    @staticmethod
    def _write_best_metric(
        path: Path,
        stage: str,
        epoch: int,
        score: Optional[float],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": stage,
                    "epoch": int(epoch),
                    "score": float(score) if score is not None else None,
                },
                handle,
                indent=2,
            )
        temporary.replace(path)

    def run(self, only_stage: Optional[str] = None) -> None:
        if only_stage is not None and only_stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage: {only_stage}")
        for stage in STAGE_NAMES:
            if only_stage is not None and stage != only_stage:
                continue
            stage_config = self.config["stages"][stage]
            epochs = int(stage_config["epochs"])
            if not bool(stage_config.get("enabled", True)) or epochs == 0:
                print(f"[exp026] skip {stage}: enabled={stage_config.get('enabled', True)}, epochs={epochs}")
                continue
            groups = configure_trainable_modules(self.model, stage)
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if not parameters:
                raise RuntimeError(f"Stage {stage} has no trainable parameters")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(stage_config["learning_rate"]),
                weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
            )
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            start_epoch = 1
            resume = stage_config.get("resume")
            if resume:
                checkpoint = load_checkpoint(
                    resume, self.model, optimizer=optimizer, map_location=self.device, strict=True
                )
                if checkpoint.get("stage") != stage:
                    raise ValueError(
                        f"Resume checkpoint stage={checkpoint.get('stage')} does not match {stage}"
                    )
                start_epoch = int(checkpoint["epoch"]) + 1
            stage_dir = self.output_root / stage
            checkpoints = stage_dir / "checkpoints"
            best_path = checkpoints / "best.pt"
            best_metric_path = stage_dir / "best_metric.json"
            if resume:
                best_loss = self._read_best_metric(best_metric_path)
            else:
                best_loss = float("inf")
                self._write_best_metric(best_metric_path, stage, 0, None)
            print(
                f"[exp026] stage={stage}, epochs={epochs}, trainable_groups={groups}, "
                f"trainable_tensors={len(trainable_parameter_names(self.model))}"
            )
            for epoch in range(start_epoch, epochs + 1):
                train_metrics = self._run_epoch(
                    stage, stage_config, self.train_loader, optimizer, scaler, epoch, True
                )
                val_metrics = None
                interval = max(1, int(self.config["training"].get("val_interval", 1)))
                if self.val_loader is not None and (epoch % interval == 0 or epoch == epochs):
                    val_metrics = self._run_epoch(
                        stage, stage_config, self.val_loader, None, scaler, epoch, False
                    )
                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                self._append_metrics(stage, record)
                score = (
                    val_metrics["loss"]
                    if val_metrics is not None
                    else (train_metrics["loss"] if self.val_loader is None else None)
                )
                if score is not None and score < best_loss:
                    best_loss = score
                    save_checkpoint(
                        best_path,
                        self.model,
                        optimizer,
                        stage,
                        epoch,
                        self.config,
                        record,
                    )
                    self._write_best_metric(best_metric_path, stage, epoch, score)
                save_checkpoint(
                    checkpoints / "last.pt",
                    self.model,
                    optimizer,
                    stage,
                    epoch,
                    self.config,
                    record,
                )
                print(f"[exp026] {stage} epoch={epoch} train={train_metrics} val={val_metrics}")

            if not best_path.exists():
                raise FileNotFoundError(
                    f"Stage {stage} finished without a best checkpoint: {best_path}"
                )
            selected = load_checkpoint(
                best_path,
                self.model,
                optimizer=None,
                map_location=self.device,
                strict=True,
            )
            print(
                f"[exp026] stage={stage} selected best.pt from epoch={selected['epoch']} "
                f"for the next stage"
            )
