from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from exp026.models.table_interaction import MissingAwareTableTokenizer
from models.global_local_mamba_patch_denoiser import ResidualMamba
from models.resunet3d import SafeGroupNorm, Swish
from models.residual_bridge_resunet3d import SinusoidalTimeEmbedding
from models.exp032g_mri_control_unet import zero_module


class ConvTimeResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, groups)
        self.act1 = Swish()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        self.norm2 = SafeGroupNorm(out_channels, groups)
        self.act2 = Swish()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        gamma, beta = torch.chunk(self.time_proj(t_emb), 2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        h = (1.0 + gamma) * self.norm2(h) + beta
        h = self.conv2(self.drop(self.act2(h)))
        return h + self.skip(x)


class TimeAwareAdaControl(nn.Module):
    def __init__(self, feature_channels: int, mri_channels: int, time_dim: int, gamma_scale: float, beta_scale: float) -> None:
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.mri_proj = nn.Conv3d(mri_channels, feature_channels * 2, 1)
        self.time_proj = nn.Linear(time_dim, feature_channels * 2)
        nn.init.zeros_(self.mri_proj.weight)
        nn.init.zeros_(self.mri_proj.bias)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, x: torch.Tensor, mri: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        raw = self.mri_proj(mri)
        t_raw = self.time_proj(t_emb)[:, :, None, None, None]
        raw = raw + t_raw
        gamma_raw, beta_raw = torch.chunk(raw, 2, dim=1)
        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        return (1.0 + gamma.to(x.dtype)) * x + beta.to(x.dtype)


class FullMRIEncoder3D(nn.Module):
    def __init__(self, channels: Sequence[int], time_dim: int, groups: int = 8, gamma_scale: float = 0.5, beta_scale: float = 0.5) -> None:
        super().__init__()
        c1, c2, c3, cb = [int(v) for v in channels]
        stem = max(16, c1 // 2)
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem, 3, stride=2, padding=1), SafeGroupNorm(stem, groups), Swish(),
            nn.Conv3d(stem, c1, 3, stride=2, padding=1), SafeGroupNorm(c1, groups), Swish(),
        )
        self.down1 = nn.Sequential(nn.Conv3d(c1, c2, 4, stride=2, padding=1), SafeGroupNorm(c2, groups), Swish())
        self.down2 = nn.Sequential(nn.Conv3d(c2, c3, 4, stride=2, padding=1), SafeGroupNorm(c3, groups), Swish())
        self.down3 = nn.Sequential(nn.Conv3d(c3, cb, 4, stride=2, padding=1), SafeGroupNorm(cb, groups), Swish())
        self.controls = nn.ModuleList([
            TimeAwareAdaControl(c1, c1, time_dim, gamma_scale, beta_scale),
            TimeAwareAdaControl(c2, c2, time_dim, gamma_scale, beta_scale),
            TimeAwareAdaControl(c3, c3, time_dim, gamma_scale, beta_scale),
            TimeAwareAdaControl(cb, cb, time_dim, gamma_scale, beta_scale),
            TimeAwareAdaControl(c3, c3, time_dim, gamma_scale, beta_scale),
            TimeAwareAdaControl(c2, c2, time_dim, gamma_scale, beta_scale),
            TimeAwareAdaControl(c1, c1, time_dim, gamma_scale, beta_scale),
        ])

    def forward(self, mri: torch.Tensor) -> list[torch.Tensor]:
        s1 = self.stem(mri)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        sb = self.down3(s3)
        return [s1, s2, s3, sb, s3, s2, s1]


class TokenGridProjector(nn.Module):
    def __init__(self, in_channels: int, global_dim: int, token_grid: Sequence[int]) -> None:
        super().__init__()
        self.token_grid = tuple(int(v) for v in token_grid)
        self.to_global = nn.Linear(in_channels, global_dim)
        self.from_global = zero_module(nn.Conv3d(global_dim, in_channels, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(x, self.token_grid)
        b, c, d, h, w = pooled.shape
        tok = pooled.view(b, c, d * h * w).transpose(1, 2)
        return self.to_global(tok)

    def decode(self, tok: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
        b, n, c = tok.shape
        gd, gh, gw = self.token_grid
        feat = tok.transpose(1, 2).reshape(b, c, gd, gh, gw)
        feat = F.interpolate(feat, size=tuple(int(v) for v in spatial_size), mode='trilinear', align_corners=False)
        return self.from_global(feat)


class BidirectionalMamba(nn.Module):
    def __init__(self, dim: int, depth: int, d_state: int = 32, d_conv: int = 4, expand: int = 2, use_checkpoint: bool = False, use_fused: bool = False) -> None:
        super().__init__()
        self.fwd = nn.ModuleList([ResidualMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand, use_fused=use_fused) for _ in range(int(depth))])
        self.bwd = nn.ModuleList([ResidualMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand, use_fused=use_fused) for _ in range(int(depth))])
        self.norm = nn.LayerNorm(dim)
        self.use_checkpoint = bool(use_checkpoint)

    def _block(self, fwd: nn.Module, bwd: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y_f = fwd(x)
        y_b = torch.flip(bwd(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (y_f + y_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fwd, bwd in zip(self.fwd, self.bwd):
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda inp, mf=fwd, mb=bwd: self._block(mf, mb, inp), x, use_reentrant=False)
            else:
                x = self._block(fwd, bwd, x)
        return self.norm(x)


class GatedResidualCrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, lambda_init: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(query_dim)
        self.kv_norm = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(query_dim, num_heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.out = zero_module(nn.Linear(query_dim, query_dim))
        self.gate = zero_module(nn.Linear(query_dim, query_dim))
        self.lambda_scale = nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.float32))
        self.last_stats: dict[str, float] = {}

    def forward(self, query: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        qn = self.q_norm(query)
        out, attn = self.attn(qn, self.kv_norm(context), self.kv_norm(context), key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        gate = torch.sigmoid(self.gate(qn))
        residual = self.lambda_scale * gate * self.out(out)
        self.last_stats = {
            'gate_mean': float(gate.detach().mean().item()),
            'gate_std': float(gate.detach().std(unbiased=False).item()),
            'lambda': float(self.lambda_scale.detach().item()),
        }
        return query + residual, attn


class ProgressiveTableState(nn.Module):
    def __init__(self, table_num_features: int, feature_type_ids: Sequence[int], table_dim: int, stage_global_dims: Sequence[int], heads: Sequence[int], time_dim: int) -> None:
        super().__init__()
        self.tokenizer = MissingAwareTableTokenizer(
            num_features=table_num_features,
            token_dim=table_dim,
            feature_type_ids=feature_type_ids,
            num_feature_types=5,
        )
        self.time_proj = nn.ModuleList([nn.Linear(time_dim, table_dim) for _ in stage_global_dims])
        self.pet_reads_table = nn.ModuleList([GatedResidualCrossAttention(gd, table_dim, h, lambda_init=0.1) for gd, h in zip(stage_global_dims, heads)])
        self.table_reads_pet = nn.ModuleList([GatedResidualCrossAttention(table_dim, gd, h, lambda_init=0.1) for gd, h in zip(stage_global_dims, heads)])

    def init_tokens(self, table: torch.Tensor, table_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok, kpm, _ = self.tokenizer(table, table_mask)
        return tok, kpm

    def stage_update(self, stage_idx: int, pet_tokens: torch.Tensor, table_tokens: torch.Tensor, table_kpm: torch.Tensor | None, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        table_tokens = table_tokens + self.time_proj[stage_idx](t_emb)[:, None, :]
        pet_tokens, attn_pt = self.pet_reads_table[stage_idx](pet_tokens, table_tokens, key_padding_mask=table_kpm)
        table_tokens, attn_tp = self.table_reads_pet[stage_idx](table_tokens, pet_tokens)
        p1 = attn_pt.detach().mean(dim=1).clamp_min(1e-8)
        p2 = attn_tp.detach().mean(dim=1).clamp_min(1e-8)
        diag = {
            'pet_table_entropy': float((-(p1 * p1.log()).sum(dim=-1).mean()).item()),
            'table_pet_entropy': float((-(p2 * p2.log()).sum(dim=-1).mean()).item()),
            'pet_table_gate_mean': float(self.pet_reads_table[stage_idx].last_stats.get('gate_mean', 0.0)),
            'pet_table_gate_std': float(self.pet_reads_table[stage_idx].last_stats.get('gate_std', 0.0)),
            'pet_table_lambda': float(self.pet_reads_table[stage_idx].last_stats.get('lambda', 0.0)),
            'table_pet_gate_mean': float(self.table_reads_pet[stage_idx].last_stats.get('gate_mean', 0.0)),
            'table_pet_gate_std': float(self.table_reads_pet[stage_idx].last_stats.get('gate_std', 0.0)),
            'table_pet_lambda': float(self.table_reads_pet[stage_idx].last_stats.get('lambda', 0.0)),
        }
        return pet_tokens, table_tokens, diag


class FusionGate(nn.Module):
    def __init__(self, channels: int, time_dim: int, lambda_init: float = 0.1) -> None:
        super().__init__()
        self.spatial_gate = nn.Conv3d(channels * 2, channels, 1)
        self.time_gate = nn.Linear(time_dim, channels)
        self.rho = nn.Sequential(nn.Linear(time_dim, channels), nn.Sigmoid())
        self.global_proj = zero_module(nn.Conv3d(channels, channels, 1))
        self.lambda_scale = nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.float32))
        self.last_stats: dict[str, float] = {}

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gproj = self.global_proj(global_feat)
        gate = torch.sigmoid(self.spatial_gate(torch.cat([local_feat, global_feat], dim=1)) + self.time_gate(t_emb)[:, :, None, None, None])
        rho = self.rho(t_emb)[:, :, None, None, None]
        self.last_stats = {
            'gate_mean': float(gate.detach().mean().item()),
            'rho_mean': float(rho.detach().mean().item()),
            'lambda': float(self.lambda_scale.detach().item()),
        }
        return local_feat + self.lambda_scale * rho.to(local_feat.dtype) * gate.to(local_feat.dtype) * gproj.to(local_feat.dtype)


class MRIStageTokenProjector(nn.Module):
    """Project MRI feature maps to global PET token space for cross-attention."""
    def __init__(self, mri_channels: int, global_dim: int, token_grid: Sequence[int]) -> None:
        super().__init__()
        self.token_grid = tuple(int(v) for v in token_grid)
        self.proj = nn.Sequential(
            nn.LayerNorm(int(mri_channels)),
            nn.Linear(int(mri_channels), int(global_dim)),
        )

    def encode(self, mri_feat: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(mri_feat, self.token_grid)
        b, c, d, h, w = pooled.shape
        tok = pooled.view(b, c, d * h * w).transpose(1, 2)
        return self.proj(tok)


class FullLatentStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, local_blocks: int, time_dim: int, groups: int, dropout: float, token_grid: Sequence[int], global_dim: int, mamba_depth: int, attn_heads: int, table_dim: int, use_checkpoint: bool, mri_channel: int = 0, use_mri_global_xattn: bool = False, inject_time_into_global: bool = False, use_fused_mamba: bool = False) -> None:
        super().__init__()
        blocks = []
        cur = in_channels
        for _ in range(int(local_blocks)):
            blocks.append(ConvTimeResBlock3D(cur, out_channels, time_dim, groups=groups, dropout=dropout))
            cur = out_channels
        self.local_blocks = nn.ModuleList(blocks)
        self.tokenizer = TokenGridProjector(out_channels, global_dim, token_grid)
        self.global_backbone = BidirectionalMamba(global_dim, mamba_depth, use_checkpoint=use_checkpoint, use_fused=use_fused_mamba)
        self.fusion = FusionGate(out_channels, time_dim)
        self.global_to_table = nn.Linear(global_dim, table_dim)
        self.table_to_global = nn.Linear(table_dim, global_dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_mri_global_xattn = bool(use_mri_global_xattn)
        self.inject_time_into_global = bool(inject_time_into_global)
        if use_mri_global_xattn and int(mri_channel) > 0:
            self.mri_token_proj = MRIStageTokenProjector(int(mri_channel), global_dim, token_grid)
            self.mri_xattn = GatedResidualCrossAttention(global_dim, global_dim, attn_heads, lambda_init=0.1)
        if inject_time_into_global:
            self.global_time_proj = nn.Linear(time_dim, global_dim)

    def _run_local(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self.local_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda a, b, m=block: m(a, b), x, t_emb, use_reentrant=False)
            else:
                x = block(x, t_emb)
        return x

    def forward(self, x: torch.Tensor, mri_feat: torch.Tensor, t_emb: torch.Tensor, control, table_state: ProgressiveTableState, table_tokens: torch.Tensor, table_kpm: torch.Tensor | None, stage_idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        local_feat = self._run_local(x, t_emb)
        local_feat = control(local_feat, mri_feat, t_emb)
        pet_tokens = self.tokenizer.encode(local_feat)
        pet_tokens = pet_tokens + 0.0 * self.global_to_table.weight.sum()
        pet_tokens, table_tokens, table_diag = table_state.stage_update(stage_idx, pet_tokens, table_tokens, table_kpm, t_emb)
        pet_tokens = pet_tokens + self.table_to_global(table_tokens.mean(dim=1, keepdim=True)).expand_as(pet_tokens)
        if self.use_mri_global_xattn and mri_feat is not None:
            mri_tokens = self.mri_token_proj.encode(mri_feat)
            pet_tokens, _ = self.mri_xattn(pet_tokens, mri_tokens)
        if self.inject_time_into_global:
            pet_tokens = pet_tokens + self.global_time_proj(t_emb)[:, None, :]
        global_tokens = self.global_backbone(pet_tokens)
        global_feat = self.tokenizer.decode(global_tokens, local_feat.shape[-3:])
        fused = self.fusion(local_feat, global_feat, t_emb)
        diag = {
            'token_count': int(global_tokens.shape[1]),
            'global_dim': int(global_tokens.shape[2]),
            'global_token_norm': float(global_tokens.detach().norm(dim=-1).mean().item()),
            **table_diag,
            **self.fusion.last_stats,
        }
        return fused, table_tokens, diag


@dataclass
class FullLatentConfig:
    image_size: tuple[int, int, int]
    latent_channels: int
    stage_channels: Sequence[int]
    stage_resblocks: Sequence[int]
    stage_token_grids: Sequence[Sequence[int]]
    stage_global_dims: Sequence[int]
    stage_mamba_depths: Sequence[int]
    stage_heads: Sequence[int]
    mri_channels: Sequence[int]
    table_dim: int
    time_dim: int
    norm_groups: int = 8
    dropout: float = 0.0
    gamma_scale: float = 0.5
    beta_scale: float = 0.5
    use_checkpoint: bool = False
    use_mri_global_xattn: bool = False
    inject_time_into_global: bool = False
    use_fused_mamba: bool = False


class FullLatentGlobalLocalUNet3D(nn.Module):
    def __init__(self, cfg: FullLatentConfig, table_num_features: int, feature_type_ids: Sequence[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_channels = int(cfg.latent_channels)
        self.stage_names = ['enc1','enc2','enc3','bottleneck','dec3','dec2','dec1']
        self.stage_spatial = [(40,48,40),(20,24,20),(10,12,10),(5,6,5),(10,12,10),(20,24,20),(40,48,40)]
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(int(cfg.time_dim)),
            nn.Linear(int(cfg.time_dim), int(cfg.time_dim) * 4),
            Swish(),
            nn.Linear(int(cfg.time_dim) * 4, int(cfg.time_dim)),
        )
        self.in_proj = nn.Conv3d(self.latent_channels, int(cfg.stage_channels[0]), 3, padding=1)
        self.mri_encoder = FullMRIEncoder3D(cfg.mri_channels[:4], int(cfg.time_dim), groups=int(cfg.norm_groups), gamma_scale=float(cfg.gamma_scale), beta_scale=float(cfg.beta_scale))
        self.stage_controls = nn.ModuleList([
            TimeAwareAdaControl(int(c), int(mc), int(cfg.time_dim), float(cfg.gamma_scale), float(cfg.beta_scale))
            for c, mc in zip(cfg.stage_channels, cfg.mri_channels)
        ])
        self.table_state = ProgressiveTableState(table_num_features, feature_type_ids, int(cfg.table_dim), cfg.stage_global_dims, cfg.stage_heads, int(cfg.time_dim))

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_c = int(cfg.stage_channels[0])
        for i in range(3):
            stage = FullLatentStage(in_c, int(cfg.stage_channels[i]), cfg.stage_resblocks[i], int(cfg.time_dim), int(cfg.norm_groups), float(cfg.dropout), cfg.stage_token_grids[i], int(cfg.stage_global_dims[i]), int(cfg.stage_mamba_depths[i]), int(cfg.stage_heads[i]), int(cfg.table_dim), bool(cfg.use_checkpoint), int(cfg.mri_channels[i]), bool(cfg.use_mri_global_xattn), bool(cfg.inject_time_into_global), bool(cfg.use_fused_mamba))
            self.enc_stages.append(stage)
            if i < 2:
                self.downs.append(nn.Conv3d(int(cfg.stage_channels[i]), int(cfg.stage_channels[i+1]), 4, stride=2, padding=1))
            else:
                self.downs.append(nn.Conv3d(int(cfg.stage_channels[i]), int(cfg.stage_channels[3]), 4, stride=2, padding=1))
            in_c = int(cfg.stage_channels[min(i+1, 3)])

        self.bottleneck = FullLatentStage(int(cfg.stage_channels[3]), int(cfg.stage_channels[3]), cfg.stage_resblocks[3], int(cfg.time_dim), int(cfg.norm_groups), float(cfg.dropout), cfg.stage_token_grids[3], int(cfg.stage_global_dims[3]), int(cfg.stage_mamba_depths[3]), int(cfg.stage_heads[3]), int(cfg.table_dim), bool(cfg.use_checkpoint), int(cfg.mri_channels[3]), bool(cfg.use_mri_global_xattn), bool(cfg.inject_time_into_global), bool(cfg.use_fused_mamba))

        self.ups = nn.ModuleList([
            nn.ConvTranspose3d(int(cfg.stage_channels[3]), int(cfg.stage_channels[4]), 4, stride=2, padding=1),
            nn.ConvTranspose3d(int(cfg.stage_channels[4]), int(cfg.stage_channels[5]), 4, stride=2, padding=1),
            nn.ConvTranspose3d(int(cfg.stage_channels[5]), int(cfg.stage_channels[6]), 4, stride=2, padding=1),
        ])
        self.dec_stages = nn.ModuleList([
            FullLatentStage(int(cfg.stage_channels[4]) + int(cfg.stage_channels[2]), int(cfg.stage_channels[4]), cfg.stage_resblocks[4], int(cfg.time_dim), int(cfg.norm_groups), float(cfg.dropout), cfg.stage_token_grids[4], int(cfg.stage_global_dims[4]), int(cfg.stage_mamba_depths[4]), int(cfg.stage_heads[4]), int(cfg.table_dim), bool(cfg.use_checkpoint), int(cfg.mri_channels[4]), bool(cfg.use_mri_global_xattn), bool(cfg.inject_time_into_global), bool(cfg.use_fused_mamba)),
            FullLatentStage(int(cfg.stage_channels[5]) + int(cfg.stage_channels[1]), int(cfg.stage_channels[5]), cfg.stage_resblocks[5], int(cfg.time_dim), int(cfg.norm_groups), float(cfg.dropout), cfg.stage_token_grids[5], int(cfg.stage_global_dims[5]), int(cfg.stage_mamba_depths[5]), int(cfg.stage_heads[5]), int(cfg.table_dim), bool(cfg.use_checkpoint), int(cfg.mri_channels[5]), bool(cfg.use_mri_global_xattn), bool(cfg.inject_time_into_global), bool(cfg.use_fused_mamba)),
            FullLatentStage(int(cfg.stage_channels[6]) + int(cfg.stage_channels[0]), int(cfg.stage_channels[6]), cfg.stage_resblocks[6], int(cfg.time_dim), int(cfg.norm_groups), float(cfg.dropout), cfg.stage_token_grids[6], int(cfg.stage_global_dims[6]), int(cfg.stage_mamba_depths[6]), int(cfg.stage_heads[6]), int(cfg.table_dim), bool(cfg.use_checkpoint), int(cfg.mri_channels[6]), bool(cfg.use_mri_global_xattn), bool(cfg.inject_time_into_global), bool(cfg.use_fused_mamba)),
        ])
        self.out_norm = SafeGroupNorm(int(cfg.stage_channels[6]), int(cfg.norm_groups))
        self.out_act = Swish()
        self.out_proj = nn.Conv3d(int(cfg.stage_channels[6]), self.latent_channels, 3, padding=1)
        self.last_stage_diagnostics: dict[str, dict[str, float]] = {}

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, mri: torch.Tensor, table: torch.Tensor, table_mask: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_encoder(t)
        mri_feats = self.mri_encoder(mri)
        table_tokens, table_kpm = self.table_state.init_tokens(table, table_mask)
        x = self.in_proj(z_t)
        skips = []
        self.last_stage_diagnostics = {}
        for idx, stage in enumerate(self.enc_stages):
            x, table_tokens, diag = stage(x, mri_feats[idx], t_emb, self.stage_controls[idx], self.table_state, table_tokens, table_kpm, idx)
            self.last_stage_diagnostics[self.stage_names[idx]] = diag
            skips.append(x)
            x = self.downs[idx](x)
        x, table_tokens, diag = self.bottleneck(x, mri_feats[3], t_emb, self.stage_controls[3], self.table_state, table_tokens, table_kpm, 3)
        self.last_stage_diagnostics['bottleneck'] = diag
        dec_indices = [(0, 2, 4), (1, 1, 5), (2, 0, 6)]
        for up_idx, skip_idx, stage_idx in dec_indices:
            x = self.ups[up_idx](x)
            skip = skips[skip_idx]
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x, table_tokens, diag = self.dec_stages[up_idx](x, mri_feats[stage_idx], t_emb, self.stage_controls[stage_idx], self.table_state, table_tokens, table_kpm, stage_idx)
            self.last_stage_diagnostics[self.stage_names[stage_idx]] = diag
        return self.out_proj(self.out_act(self.out_norm(x)))
