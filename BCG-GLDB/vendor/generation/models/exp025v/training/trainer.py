from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp025v.losses.amrf import AMRFLoss
from exp025v.models.exp025v_model import Exp025VMRI2PET
from exp025u.training.trainer import Exp025UTrainer
from exp025v.training.stages import (
    configure_trainable_modules,
    set_stage_mode,
)
from metrics.image_metrics import compute_image_metrics_detailed


class Exp025VTrainer(Exp025UTrainer):
    """Exp025u model optimized with the MRI-to-PET-specific AMRF loss."""

    model: Exp025VMRI2PET

    def __init__(
        self,
        model: Exp025VMRI2PET,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device,
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device)
        loss_config = dict(
            config["training"].get("innovative_mri_pet_loss", {})
        )
        loss_name = loss_config.pop("name", "amrf")
        if loss_name != "amrf":
            raise ValueError(f"Unsupported MRI-to-PET loss: {loss_name}")
        self.amrf = AMRFLoss(
            data_range=self.metric_data_range,
            **loss_config,
        ).to(device)

    def _configure_trainable_modules(self, stage: str) -> list[str]:
        return configure_trainable_modules(self.model, stage)

    def _set_stage_mode(self, stage: str, training: bool) -> None:
        set_stage_mode(self.model, stage, training)

    def _image_objective_v(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mri: torch.Tensor,
        brain_mask: torch.Tensor,
        atlas: torch.Tensor,
        refined_latent: torch.Tensor,
        target_latent: Optional[torch.Tensor],
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction32 = prediction.float()
        target32 = target.float()
        mri32 = mri.float()
        mask32 = brain_mask.float()
        atlas32 = atlas.float()

        with torch.autocast(
            device_type=prediction32.device.type, enabled=False
        ):
            amrf_loss, amrf_metrics = self.amrf(
                prediction32, target32, mri32, mask32, atlas32
            )
        if target_latent is None:
            latent = prediction32.new_zeros(())
        else:
            latent = F.smooth_l1_loss(
                refined_latent.float(), target_latent.float()
            )
        range_penalty = F.relu(prediction32.abs() - 1.0).square().mean()
        loss = (
            self._weight(stage_config, "amrf", 1.0) * amrf_loss
            + self._weight(stage_config, "latent", 0.0) * latent
            + self._weight(stage_config, "range", 0.05) * range_penalty
        )
        metrics = {
            "loss": loss.detach(),
            "l1": F.l1_loss(prediction32, target32).detach(),
            "brain_l1": self._masked_l1(
                prediction32, target32, mask32
            ).detach(),
            "latent": latent.detach(),
            "range": range_penalty.detach(),
            "smooth_weight": self.model.smooth_weight.detach(),
            "atlas_weight": self.model.atlas_weight.detach(),
            "coarse_latent_weight": torch.sigmoid(
                self.model.coarse_gate_logit.detach()
            ),
            "direct_residual_scale": (
                self.model.direct_residual_scale.detach()
            ),
            "post_atlas_weight": self.model.post_atlas_weight.detach(),
            "residual_prior_strength": (
                self.model.residual_prior_strength.detach()
            ),
            "post_smooth_weight": self.model.post_smooth_weight.detach(),
        }
        metrics.update(amrf_metrics)
        return loss, metrics

    def _direct_step(
        self,
        stage: str,
        batch: dict,
        stage_config: dict,
        include_official_metrics: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        target_latent = None
        if self._weight(stage_config, "latent", 0.0) > 0:
            with torch.no_grad():
                target_latent = self.model.encode_pet_patches(
                    pet,
                    self.patch_batch_size,
                    sample=False,
                    encoder_grad=False,
                )
        loss, metrics = self._image_objective_v(
            prediction,
            pet,
            mri,
            brain_mask,
            auxiliary["atlas_pet"],
            auxiliary["refined_latent"],
            target_latent,
            stage_config,
        )
        if include_official_metrics:
            with torch.autocast(
                device_type=prediction.device.type, enabled=False
            ):
                official = compute_image_metrics_detailed(
                    prediction.detach().float().clamp(-1.0, 1.0),
                    pet.detach().float(),
                    mask=brain_mask.detach().float(),
                    data_range=self.metric_data_range,
                )
            for name, value in official.items():
                metrics[name] = prediction.new_tensor(float(value))
            metrics["selection_score"] = prediction.new_tensor(
                self._selection_score(official)
            )
        return loss, metrics
