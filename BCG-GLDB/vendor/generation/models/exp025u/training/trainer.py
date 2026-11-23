from __future__ import annotations

from typing import Optional

import torch

from exp025t.training.trainer import Exp025TTrainer
from exp025u.models.exp025u_model import Exp025UMRI2PET
from exp025u.training.stages import (
    configure_trainable_modules,
    set_stage_mode,
)


class Exp025UTrainer(Exp025TTrainer):
    """Exp025t objective with Exp025u spatial-calibration controls."""

    model: Exp025UMRI2PET

    def _configure_trainable_modules(self, stage: str) -> list[str]:
        return configure_trainable_modules(self.model, stage)

    def _set_stage_mode(self, stage: str, training: bool) -> None:
        set_stage_mode(self.model, stage, training)

    def _image_objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor,
        refined_latent: torch.Tensor,
        target_latent: Optional[torch.Tensor],
        stage_config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = super()._image_objective(
            prediction,
            target,
            brain_mask,
            refined_latent,
            target_latent,
            stage_config,
        )
        metrics.update(
            {
                "post_atlas_weight": self.model.post_atlas_weight.detach(),
                "residual_prior_strength": (
                    self.model.residual_prior_strength.detach()
                ),
                "post_smooth_weight": (
                    self.model.post_smooth_weight.detach()
                ),
            }
        )
        return loss, metrics
