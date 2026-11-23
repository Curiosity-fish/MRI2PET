"""Exp033d: frozen BSFE MRI encoder + Ada (TimeAwareAdaControl) bridge denoiser.

Design (user request):
- Based on Exp033c local U-Net backbone (FullLatentGlobalLocalUNet3D local path),
  WITHOUT global token/Mamba branch and WITHOUT table conditioning.
- MRI guidance: pretrained/frozen BSFE (Exp011b) acts as the MRI encoder.
  Its multi-scale pyramid features are injected into every U-Net stage through the
  original Ada form (TimeAwareAdaControl: gamma/beta modulation, time-aware).
- The frozen BSFE head predicts a coarse 1-channel PET latent that is used as the
  bridge condition (projected to `latent_channels` by a small trainable 1x1 conv),
  mirroring Exp025c's residual bridge (T=100, 1-step sampling).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.full_latent_gl_model import ConvTimeResBlock3D, TimeAwareAdaControl
from models.residual_bridge_resunet3d import SinusoidalTimeEmbedding
from models.resunet3d import SafeGroupNorm, Swish


class LocalStage(nn.Module):
    """Stack of local ConvTimeResBlock3D blocks (no global/token/table)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        local_blocks: int,
        time_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        blocks = []
        cur = int(in_channels)
        for _ in range(int(local_blocks)):
            blocks.append(ConvTimeResBlock3D(cur, int(out_channels), int(time_dim), groups=groups, dropout=dropout))
            cur = int(out_channels)
        self.local_blocks = nn.ModuleList(blocks)
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self.local_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda a, b, m=block: m(a, b), x, t_emb, use_reentrant=False)
            else:
                x = block(x, t_emb)
        return x


class BSFEAdaBridgeUNet3D(nn.Module):
    """7-stage local U-Net (Exp033c local path) with BSFE-pyramid Ada injection.

    Input: x_t (B,C,latent res) and bridge condition (B,1,latent res).
    Output: residual prediction (B,C,latent res), pet_pred = x_t - residual.
    """

    def __init__(
        self,
        latent_channels: int,
        stage_channels: Sequence[int],
        stage_resblocks: Sequence[int],
        mri_channels: Sequence[int],
        time_dim: int,
        norm_groups: int = 8,
        dropout: float = 0.0,
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        sc = [int(c) for c in stage_channels]
        mc = [int(c) for c in mri_channels]
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(time_dim) * 4),
            Swish(),
            nn.Linear(int(time_dim) * 4, int(time_dim)),
        )
        # 1ch BSFE head latent_prediction -> latent_channels (bridge condition).
        self.cond_proj = nn.Conv3d(1, self.latent_channels, 1)
        self.in_proj = nn.Conv3d(self.latent_channels * 2, sc[0], 3, padding=1)
        # Ada controls: BSFE pyramid levels [48,96,192,384,192,96,48] -> stage channels.
        self.stage_controls = nn.ModuleList([
            TimeAwareAdaControl(int(c), int(m), int(time_dim), float(gamma_scale), float(beta_scale))
            for c, m in zip(sc, mc)
        ])

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_c = sc[0]
        for i in range(3):
            self.enc_stages.append(LocalStage(in_c, sc[i], stage_resblocks[i], time_dim, norm_groups, dropout, use_checkpoint))
            self.downs.append(nn.Conv3d(sc[i], sc[i + 1], 4, stride=2, padding=1))
            in_c = sc[i + 1]
        self.bottleneck = LocalStage(sc[3], sc[3], stage_resblocks[3], time_dim, norm_groups, dropout, use_checkpoint)
        self.ups = nn.ModuleList([
            nn.ConvTranspose3d(sc[3], sc[4], 4, stride=2, padding=1),
            nn.ConvTranspose3d(sc[4], sc[5], 4, stride=2, padding=1),
            nn.ConvTranspose3d(sc[5], sc[6], 4, stride=2, padding=1),
        ])
        self.dec_stages = nn.ModuleList([
            LocalStage(sc[4] + sc[2], sc[4], stage_resblocks[4], time_dim, norm_groups, dropout, use_checkpoint),
            LocalStage(sc[5] + sc[1], sc[5], stage_resblocks[5], time_dim, norm_groups, dropout, use_checkpoint),
            LocalStage(sc[6] + sc[0], sc[6], stage_resblocks[6], time_dim, norm_groups, dropout, use_checkpoint),
        ])
        self.out_norm = SafeGroupNorm(sc[6], int(norm_groups))
        self.out_act = Swish()
        self.out_proj = nn.Conv3d(sc[6], self.latent_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t_emb: torch.Tensor, pyramid) -> torch.Tensor:
        x = self.in_proj(torch.cat([x_t, condition], dim=1))
        # BSFE pyramid has 4 encoder levels [48,96,192,384] at latent resolutions
        # (40,48,40)/(20,24,20)/(10,12,10)/(5,6,5). Mirror them for the 7 U-Net
        # stages: enc 0,1,2 use pyramid 0,1,2; bottleneck uses pyramid 3;
        # dec 4,5,6 reuse pyramid 2,1,0 (resolution matches each decoder stage).
        ada_feats = [pyramid[0], pyramid[1], pyramid[2], pyramid[3], pyramid[2], pyramid[1], pyramid[0]]
        skips = []
        for i in range(3):
            x = self.enc_stages[i](x, t_emb)
            x = self.stage_controls[i](x, ada_feats[i], t_emb)
            skips.append(x)
            x = self.downs[i](x)
        x = self.bottleneck(x, t_emb)
        x = self.stage_controls[3](x, ada_feats[3], t_emb)
        for up_idx, skip_idx, stage_idx in [(0, 2, 4), (1, 1, 5), (2, 0, 6)]:
            x = self.ups[up_idx](x)
            skip = skips[skip_idx]
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_stages[up_idx](x, t_emb)
            x = self.stage_controls[stage_idx](x, ada_feats[stage_idx], t_emb)
        return self.out_proj(self.out_act(self.out_norm(x)))


class BSFEAdaBridgeModel(nn.Module):
    """Frozen BSFE(+head) condition encoder + trainable BSFEAdaBridgeUNet3D.

    Compatible with ResidualBridgeDiffusion.model(x_t, condition, t, **model_kwargs)
    where model_kwargs={'mri': mri}.
    """

    def __init__(self, bsfe: nn.Module, bsfe_head: nn.Module, unet: BSFEAdaBridgeUNet3D) -> None:
        super().__init__()
        self.bsfe = bsfe
        self.bsfe_head = bsfe_head
        self.unet = unet
        # By default the MRI encoder (BSFE) and its head are frozen; only the
        # U-Net is trainable (Exp033d / Exp033e phase-1 behavior).
        self._bsfe_trainable = False
        self.freeze_bsfe()

    # -- BSFE freeze/unfreeze management (Exp033e two-stage) ----------------
    def freeze_bsfe(self) -> None:
        self._bsfe_trainable = False
        for module in (self.bsfe, self.bsfe_head):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def unfreeze_bsfe(self) -> None:
        self._bsfe_trainable = True
        for module in (self.bsfe, self.bsfe_head):
            module.train()
            for p in module.parameters():
                p.requires_grad_(True)

    def train(self, mode: bool = True):
        # Keep a frozen BSFE (+head) in eval mode even when the U-Net is in
        # train mode, so frozen dropout/GN statistics never inject stochasticity.
        super().train(mode)
        if not self._bsfe_trainable:
            self.bsfe.eval()
            self.bsfe_head.eval()
        return self

    @torch.no_grad()
    def encode_condition(self, mri: torch.Tensor):
        out = self.bsfe(mri)
        head_out = self.bsfe_head(out['pyramid'], out['global'], out['tokens'])
        return head_out['latent_prediction'], head_out['pet_prediction'], out['pyramid']

    def encode_condition_train(self, mri: torch.Tensor):
        """Trainable version of encode_condition() (phase-2 gradient flow)."""
        out = self.bsfe(mri)
        head_out = self.bsfe_head(out['pyramid'], out['global'], out['tokens'])
        return head_out['latent_prediction'], head_out['pet_prediction'], out['pyramid']

    def encode_pyramid_train(self, mri: torch.Tensor):
        """Trainable BSFE encoder forward returning only the pyramid (phase 2).

        The bridge condition is served from the frozen-BSFE cache, so only the
        pyramid feeds the Ada controls and carries gradients back to BSFE.
        """
        return self.bsfe(mri)['pyramid']

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor, pyramid) -> torch.Tensor:
        # `condition` (coarse PET latent, same space as x_t) and `pyramid` are
        # precomputed by the caller via encode_condition(), so BSFE runs once/step.
        t_emb = self.unet.time_encoder(t)
        if condition.shape[1] == 1:
            cond = self.unet.cond_proj(condition)
        else:
            cond = condition
        return self.unet(x_t, cond, t_emb, pyramid)
