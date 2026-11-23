"""Exp033h: Exp033e two-stage BSFE + Ada bridge denoiser + table guidance.

Brings Exp033a's table-guidance idea onto the Exp033e backbone:
  - missing-aware table tokenizer (25 ADNI features, standardized, missing mask)
  - ProgressiveTableState: per-stage bidirectional gated-residual cross-attention
    between PET global tokens and table tokens (time-aware), + subject-level
    table summary injected into every PET token (lambda init 0.1)
  - per-stage TokenGridProjector -> (table cross-attn) -> decode ->
    time-gated FusionGate (lambda init 0.1) fused back into local U-Net features

The bridge conditioning / Ada pyramid path is identical to Exp033e:
  - condition = frozen-BSFE pet_prediction -> patch AE (4ch latent), cached in stage 2
  - pyramid (BSFE) -> TimeAwareAdaControl injection at every stage
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.full_latent_gl_model import (
    BidirectionalMamba,
    ConvTimeResBlock3D,
    FusionGate,
    ProgressiveTableState,
    TimeAwareAdaControl,
    TokenGridProjector,
)
from models.residual_bridge_resunet3d import SinusoidalTimeEmbedding
from models.resunet3d import SafeGroupNorm, Swish


class BSFEAdaTableStage(nn.Module):
    """Local conv stage + BSFE-pyramid Ada control + table-guided global tokens.

    local -> Ada(MRI pyramid) -> PET tokens -> bidirectional table cross-attn
    -> table summary add -> (optional Mamba) -> decode -> FusionGate back into local.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        local_blocks: int,
        time_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
        token_grid: Sequence[int] = (8, 8, 8),
        global_dim: int = 232,
        mamba_depth: int = 0,
        attn_heads: int = 8,
        table_dim: int = 128,
        use_checkpoint: bool = False,
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        use_fused_mamba: bool = False,
        table_lambda_init: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        cur = int(in_channels)
        for _ in range(int(local_blocks)):
            blocks.append(ConvTimeResBlock3D(cur, int(out_channels), int(time_dim), groups=groups, dropout=dropout))
            cur = int(out_channels)
        self.local_blocks = nn.ModuleList(blocks)
        self.tokenizer = TokenGridProjector(int(out_channels), int(global_dim), token_grid)
        if int(mamba_depth) > 0:
            self.global_backbone = BidirectionalMamba(int(global_dim), int(mamba_depth), use_checkpoint=bool(use_checkpoint), use_fused=bool(use_fused_mamba))
        else:
            self.global_backbone = None
        self.table_to_global = nn.Linear(int(table_dim), int(global_dim))
        self.fusion = FusionGate(int(out_channels), int(time_dim), lambda_init=float(table_lambda_init))
        self.use_checkpoint = bool(use_checkpoint)
        self.global_dim = int(global_dim)

    def _run_local(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self.local_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda a, b, m=block: m(a, b), x, t_emb, use_reentrant=False)
            else:
                x = block(x, t_emb)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mri_feat: torch.Tensor,
        t_emb: torch.Tensor,
        control,
        table_state: ProgressiveTableState,
        table_tokens: torch.Tensor,
        table_kpm: torch.Tensor | None,
        stage_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        local_feat = self._run_local(x, t_emb)
        local_feat = control(local_feat, mri_feat, t_emb)
        pet_tokens = self.tokenizer.encode(local_feat)
        pet_tokens, table_tokens, table_diag = table_state.stage_update(
            stage_idx, pet_tokens, table_tokens, table_kpm, t_emb
        )
        # subject-level table summary -> added to every PET token (Exp033a idea)
        pet_tokens = pet_tokens + self.table_to_global(table_tokens.mean(dim=1, keepdim=True)).expand_as(pet_tokens)
        if self.global_backbone is not None:
            pet_tokens = self.global_backbone(pet_tokens)
        global_feat = self.tokenizer.decode(pet_tokens, local_feat.shape[-3:])
        fused = self.fusion(local_feat, global_feat, t_emb)
        diag = {
            'token_count': int(pet_tokens.shape[1]),
            'global_dim': self.global_dim,
            'global_token_norm': float(pet_tokens.detach().norm(dim=-1).mean().item()),
            **table_diag,
            **self.fusion.last_stats,
        }
        return fused, table_tokens, diag


class BSFEAdaTableBridgeUNet3D(nn.Module):
    """7-stage local U-Net (Exp033e) + Exp033a-style table-guidance branch.

    Input: x_t (B,C,latent res), bridge condition (B,C,latent res), t_emb,
    BSFE pyramid (list of 7 levels), table (B,F), table_mask (B,F).
    Output: residual prediction (B,C,latent res).
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
        table_num_features: int = 0,
        feature_type_ids: Sequence[int] | None = None,
        table_dim: int = 128,
        stage_token_grids: Sequence[Sequence[int]] | None = None,
        stage_global_dims: Sequence[int] | None = None,
        stage_heads: Sequence[int] | None = None,
        stage_mamba_depths: Sequence[int] | None = None,
        use_fused_mamba: bool = False,
        table_lambda_init: float = 0.1,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        sc = [int(c) for c in stage_channels]
        mc = [int(c) for c in mri_channels]
        n_stages = len(sc)
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
        # Table guidance (Exp033a).
        gdims = list(stage_global_dims) if stage_global_dims is not None else [int(c) for c in sc]
        grids = list(stage_token_grids) if stage_token_grids is not None else [(8, 8, 8)] * n_stages
        heads = list(stage_heads) if stage_heads is not None else [8] * n_stages
        mdepth = list(stage_mamba_depths) if stage_mamba_depths is not None else [0] * n_stages
        if len(gdims) != n_stages or len(grids) != n_stages or len(heads) != n_stages or len(mdepth) != n_stages:
            raise ValueError('stage_token_grids/global_dims/heads/mamba_depths must match stage_channels length')
        self.table_state = ProgressiveTableState(
            table_num_features=int(table_num_features),
            feature_type_ids=list(feature_type_ids) if feature_type_ids is not None else None,
            table_dim=int(table_dim),
            stage_global_dims=gdims,
            heads=heads,
            time_dim=int(time_dim),
        )

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_c = sc[0]
        for i in range(3):
            self.enc_stages.append(BSFEAdaTableStage(in_c, sc[i], stage_resblocks[i], time_dim, norm_groups, dropout,
                                                     grids[i], gdims[i], mdepth[i], heads[i], table_dim, use_checkpoint,
                                                     gamma_scale, beta_scale, use_fused_mamba, table_lambda_init))
            self.downs.append(nn.Conv3d(sc[i], sc[i + 1], 4, stride=2, padding=1))
            in_c = sc[i + 1]
        self.bottleneck = BSFEAdaTableStage(sc[3], sc[3], stage_resblocks[3], time_dim, norm_groups, dropout,
                                            grids[3], gdims[3], mdepth[3], heads[3], table_dim, use_checkpoint,
                                            gamma_scale, beta_scale, use_fused_mamba, table_lambda_init)
        self.ups = nn.ModuleList([
            nn.ConvTranspose3d(sc[3], sc[4], 4, stride=2, padding=1),
            nn.ConvTranspose3d(sc[4], sc[5], 4, stride=2, padding=1),
            nn.ConvTranspose3d(sc[5], sc[6], 4, stride=2, padding=1),
        ])
        self.dec_stages = nn.ModuleList([
            BSFEAdaTableStage(sc[4] + sc[2], sc[4], stage_resblocks[4], time_dim, norm_groups, dropout,
                              grids[4], gdims[4], mdepth[4], heads[4], table_dim, use_checkpoint,
                              gamma_scale, beta_scale, use_fused_mamba, table_lambda_init),
            BSFEAdaTableStage(sc[5] + sc[1], sc[5], stage_resblocks[5], time_dim, norm_groups, dropout,
                              grids[5], gdims[5], mdepth[5], heads[5], table_dim, use_checkpoint,
                              gamma_scale, beta_scale, use_fused_mamba, table_lambda_init),
            BSFEAdaTableStage(sc[6] + sc[0], sc[6], stage_resblocks[6], time_dim, norm_groups, dropout,
                              grids[6], gdims[6], mdepth[6], heads[6], table_dim, use_checkpoint,
                              gamma_scale, beta_scale, use_fused_mamba, table_lambda_init),
        ])
        self.out_norm = SafeGroupNorm(sc[6], int(norm_groups))
        self.out_act = Swish()
        self.out_proj = nn.Conv3d(sc[6], self.latent_channels, 3, padding=1)
        self.zero_init_residual = bool(zero_init_residual)
        if self.zero_init_residual:
            self.apply_zero_init_residual()
        self.stage_names = [f'enc{i+1}' for i in range(3)] + ['bottleneck'] + [f'dec{i+1}' for i in range(3)]
        self.last_stage_diagnostics: dict[str, dict[str, float]] = {}

    def apply_zero_init_residual(self) -> None:
        """Zero-init the final residual head so the initial output is 0
        (i.e. pet_pred = x_t, a 'copy-the-condition' starting point for
        one-step bridge sampling)."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.zero_init_residual = True

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t_emb: torch.Tensor,
        pyramid,
        table: torch.Tensor,
        table_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_proj(torch.cat([x_t, condition], dim=1))
        # BSFE pyramid has 4 encoder levels [48,96,192,384] at latent resolutions
        # (40,48,40)/(20,24,20)/(10,12,10)/(5,6,5); mirror for 7 U-Net stages.
        ada_feats = [pyramid[0], pyramid[1], pyramid[2], pyramid[3], pyramid[2], pyramid[1], pyramid[0]]
        table_tokens, table_kpm = self.table_state.init_tokens(table, table_mask)
        skips = []
        self.last_stage_diagnostics = {}
        for i in range(3):
            x, table_tokens, diag = self.enc_stages[i](x, ada_feats[i], t_emb, self.stage_controls[i],
                                                       self.table_state, table_tokens, table_kpm, i)
            self.last_stage_diagnostics[self.stage_names[i]] = diag
            skips.append(x)
            x = self.downs[i](x)
        x, table_tokens, diag = self.bottleneck(x, ada_feats[3], t_emb, self.stage_controls[3],
                                                self.table_state, table_tokens, table_kpm, 3)
        self.last_stage_diagnostics['bottleneck'] = diag
        for up_idx, skip_idx, stage_idx in [(0, 2, 4), (1, 1, 5), (2, 0, 6)]:
            x = self.ups[up_idx](x)
            skip = skips[skip_idx]
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x, table_tokens, diag = self.dec_stages[up_idx](x, ada_feats[stage_idx], t_emb, self.stage_controls[stage_idx],
                                                            self.table_state, table_tokens, table_kpm, stage_idx)
            self.last_stage_diagnostics[self.stage_names[stage_idx]] = diag
        return self.out_proj(self.out_act(self.out_norm(x)))


class BSFEAdaTableBridgeModel(nn.Module):
    """BSFE(+head) condition encoder + BSFEAdaTableBridgeUNet3D.

    Compatible with ResidualBridgeDiffusion.model(x_t, condition, t, **model_kwargs)
    where model_kwargs={'pyramid': pyramid, 'table': table, 'table_mask': table_mask}.
    """

    def __init__(self, bsfe: nn.Module, bsfe_head: nn.Module, unet: BSFEAdaTableBridgeUNet3D) -> None:
        super().__init__()
        self.bsfe = bsfe
        self.bsfe_head = bsfe_head
        self.unet = unet
        self._bsfe_trainable = False
        self.freeze_bsfe()

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
        super().train(mode)
        if mode:
            if not self._bsfe_trainable:
                self.bsfe.eval()
                self.bsfe_head.eval()
        else:
            self.bsfe.eval()
            self.bsfe_head.eval()
        return self

    @torch.no_grad()
    def encode_condition(self, mri: torch.Tensor):
        out = self.bsfe(mri)
        head_out = self.bsfe_head(out['pyramid'], out['global'], out['tokens'])
        return head_out['latent_prediction'], head_out['pet_prediction'], out['pyramid']

    def encode_pyramid_train(self, mri: torch.Tensor):
        return self.bsfe(mri)['pyramid']

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        pyramid,
        table: torch.Tensor,
        table_mask: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.unet.time_encoder(t)
        if condition.shape[1] == 1:
            cond = self.unet.cond_proj(condition)
        else:
            cond = condition
        return self.unet(x_t, cond, t_emb, pyramid, table, table_mask)
