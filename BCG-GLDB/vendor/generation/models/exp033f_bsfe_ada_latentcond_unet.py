"""Exp033f: BSFE + Ada bridge denoiser with latent-MRI bridge condition.

Difference vs Exp033d/Exp033e:
- The bridge condition is NOT the BSFE-head-predicted PET (pet_prediction ->
  patch AE). Instead it is built from the BSFE encoder's pyramid[0] -- the
  highest-resolution multi-scale MRI feature, which lives at the SAME spatial
  scale as the 4ch patch-AE latent (image/4) -- through a small trainable
  zero-init 1x1 conv (48ch -> latent_channels). The BSFE head is not used.
- Everything else matches Exp033e: 7-stage local U-Net (Exp033c local path),
  TimeAwareAdaControl injection of the full BSFE pyramid at every stage,
  residual bridge T=100, 1-step eval, no global/table.
- Two-stage schedule: epochs 1..freeze: BSFE frozen (train unet + cond_proj);
  epochs freeze+1..end: BSFE trainable (gradient flows both through the Ada
  pyramid and through cond_proj(pyramid[0])).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.exp032g_mri_control_unet import zero_module
from models.exp033d_bsfe_ada_bridge_unet import BSFEAdaBridgeUNet3D


class BSFEAdaLatentCondModel(nn.Module):
    """BSFE encoder + trainable latent-MRI condition projector + Ada U-Net.

    Compatible with ResidualBridgeDiffusion.model(x_t, condition, t, **kwargs)
    where model_kwargs={'pyramid': pyramid} and `condition` is the 4ch latent
    MRI condition produced by cond_from_pyramid().
    """

    def __init__(
        self,
        bsfe: nn.Module,
        unet: BSFEAdaBridgeUNet3D,
        cond_in_channels: int,
        latent_channels: int,
    ) -> None:
        super().__init__()
        self.bsfe = bsfe
        self.unet = unet
        # pyramid[0] (same-scale latent MRI feature) -> 4ch bridge condition.
        # Zero-init so the condition starts as 0 and is learned (controlled
        # injection, consistent with the Ada zero-init style).
        self.cond_proj = zero_module(nn.Conv3d(int(cond_in_channels), int(latent_channels), 1))
        self._bsfe_trainable = False
        self.freeze_bsfe()

    # -- BSFE freeze/unfreeze management ------------------------------------
    def freeze_bsfe(self) -> None:
        self._bsfe_trainable = False
        self.bsfe.eval()
        for p in self.bsfe.parameters():
            p.requires_grad_(False)

    def unfreeze_bsfe(self) -> None:
        self._bsfe_trainable = True
        self.bsfe.train()
        for p in self.bsfe.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True):
        # Keep a frozen BSFE in eval mode even when the U-Net is in train mode.
        super().train(mode)
        if not self._bsfe_trainable:
            self.bsfe.eval()
        return self

    @torch.no_grad()
    def encode_pyramid(self, mri: torch.Tensor):
        """Frozen BSFE forward -> multi-scale pyramid (no gradient)."""
        return self.bsfe(mri)['pyramid']

    def encode_pyramid_train(self, mri: torch.Tensor):
        """Trainable BSFE forward -> multi-scale pyramid (phase-2 gradient)."""
        return self.bsfe(mri)['pyramid']

    def cond_from_pyramid(self, pyramid) -> torch.Tensor:
        """Build the 4ch latent bridge condition from the same-scale MRI feature."""
        return self.cond_proj(pyramid[0])

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor, pyramid) -> torch.Tensor:
        # `condition` is already a 4ch latent (cond_from_pyramid output).
        t_emb = self.unet.time_encoder(t)
        return self.unet(x_t, condition, t_emb, pyramid)
