from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.exp026.models.table_interaction import MissingAwareTableTokenizer
from models.exp032g_mri_control_unet import zero_module
from models.exp032h_global_local_unet import Exp032HGlobalLocalUNet3D
from models.residual_bridge_resunet3d import ResidualTimeBlock3D


class PatchTokenProjector(nn.Module):
    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x.mean(dim=(-1, -2, -3))))


class ZeroInitCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.q = nn.LayerNorm(dim)
        self.kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out = zero_module(nn.Linear(dim, dim))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, attn = self.attn(
            self.q(query),
            self.kv(context),
            self.kv(context),
            key_padding_mask=context_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return self.out(out), attn


class TokenToModulation(nn.Module):
    def __init__(self, token_dim: int, feature_channels: int, gamma_scale: float, beta_scale: float) -> None:
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.gamma = zero_module(nn.Linear(token_dim, feature_channels))
        self.beta = zero_module(nn.Linear(token_dim, feature_channels))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_scale * torch.tanh(self.gamma(tokens))
        beta = self.beta_scale * torch.tanh(self.beta(tokens))
        return gamma, beta


@dataclass
class TableFusionOutput:
    pet_tokens: torch.Tensor
    table_tokens: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    diagnostics: dict[str, float]


class PETBiCATableFusion(nn.Module):
    def __init__(self, token_dim: int, feature_channels: int, table_num_features: int, feature_type_ids: Sequence[int], gamma_scale: float, beta_scale: float, num_heads: int = 4) -> None:
        super().__init__()
        self.table = MissingAwareTableTokenizer(
            num_features=table_num_features,
            token_dim=token_dim,
            feature_type_ids=feature_type_ids,
            num_feature_types=5,
        )
        self.pet_reads_table = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.table_reads_pet = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.pet_mod = TokenToModulation(token_dim, feature_channels, gamma_scale, beta_scale)

    @staticmethod
    def _entropy(attn: torch.Tensor) -> float:
        p = attn.detach().mean(dim=1).clamp_min(1e-8)
        return float((-(p * p.log()).sum(dim=-1).mean()).item())

    def forward(self, pet_tokens: torch.Tensor, _mri_tokens: torch.Tensor, table: torch.Tensor, table_mask: torch.Tensor) -> TableFusionOutput:
        table_tokens, table_kpm, _ = self.table(table, table_mask)
        dt, attn_pt = self.pet_reads_table(pet_tokens, table_tokens, context_key_padding_mask=table_kpm)
        pet_tokens = pet_tokens + dt
        dp, attn_tp = self.table_reads_pet(table_tokens, pet_tokens)
        table_tokens = table_tokens + dp
        gamma, beta = self.pet_mod(pet_tokens)
        return TableFusionOutput(
            pet_tokens=pet_tokens,
            table_tokens=table_tokens,
            gamma=gamma,
            beta=beta,
            diagnostics={
                "pet_table_attn_entropy": self._entropy(attn_pt),
                "table_pet_attn_entropy": self._entropy(attn_tp),
                "gamma_mean": float(gamma.detach().mean().item()),
                "gamma_std": float(gamma.detach().std(unbiased=False).item()),
                "beta_mean": float(beta.detach().mean().item()),
                "beta_std": float(beta.detach().std(unbiased=False).item()),
            },
        )


class DualBiCATableFusion(nn.Module):
    def __init__(self, token_dim: int, feature_channels: int, table_num_features: int, feature_type_ids: Sequence[int], gamma_scale: float, beta_scale: float, num_heads: int = 4) -> None:
        super().__init__()
        self.table = MissingAwareTableTokenizer(
            num_features=table_num_features,
            token_dim=token_dim,
            feature_type_ids=feature_type_ids,
            num_feature_types=5,
        )
        self.table_reads_mri = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.mri_reads_table = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.pet_reads_table = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.table_reads_pet = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.pet_mod = TokenToModulation(token_dim, feature_channels, gamma_scale, beta_scale)

    @staticmethod
    def _entropy(attn: torch.Tensor) -> float:
        p = attn.detach().mean(dim=1).clamp_min(1e-8)
        return float((-(p * p.log()).sum(dim=-1).mean()).item())

    def forward(self, pet_tokens: torch.Tensor, mri_tokens: torch.Tensor, table: torch.Tensor, table_mask: torch.Tensor) -> TableFusionOutput:
        table_tokens, table_kpm, _ = self.table(table, table_mask)
        dtm, attn_tm = self.table_reads_mri(table_tokens, mri_tokens)
        table_mri = table_tokens + dtm
        dmt, attn_mt = self.mri_reads_table(mri_tokens, table_mri, context_key_padding_mask=table_kpm)
        _mri_table = mri_tokens + dmt
        dpt, attn_pt = self.pet_reads_table(pet_tokens, table_mri, context_key_padding_mask=table_kpm)
        pet_tokens = pet_tokens + dpt
        dtp, attn_tp = self.table_reads_pet(table_mri, pet_tokens)
        table_final = table_mri + dtp
        gamma, beta = self.pet_mod(pet_tokens)
        return TableFusionOutput(
            pet_tokens=pet_tokens,
            table_tokens=table_final,
            gamma=gamma,
            beta=beta,
            diagnostics={
                "table_mri_attn_entropy": self._entropy(attn_tm),
                "mri_table_attn_entropy": self._entropy(attn_mt),
                "pet_table_attn_entropy": self._entropy(attn_pt),
                "table_pet_attn_entropy": self._entropy(attn_tp),
                "gamma_mean": float(gamma.detach().mean().item()),
                "gamma_std": float(gamma.detach().std(unbiased=False).item()),
                "beta_mean": float(beta.detach().mean().item()),
                "beta_std": float(beta.detach().std(unbiased=False).item()),
            },
        )


class StructRouteTableFusion(nn.Module):
    def __init__(self, token_dim: int, feature_channels: int, table_num_features: int, feature_type_ids: Sequence[int], gamma_scale: float, beta_scale: float, routing_temperature: float = 1.0, num_heads: int = 4) -> None:
        super().__init__()
        self.routing_temperature = float(routing_temperature)
        self.table = MissingAwareTableTokenizer(
            num_features=table_num_features,
            token_dim=token_dim,
            feature_type_ids=feature_type_ids,
            num_feature_types=5,
        )
        self.table_reads_mri = ZeroInitCrossAttention(token_dim, num_heads=num_heads)
        self.q = nn.Linear(token_dim, token_dim)
        self.k = nn.Linear(token_dim, token_dim)
        self.v = nn.Linear(token_dim, token_dim)
        self.pet_mod = TokenToModulation(token_dim, feature_channels, gamma_scale, beta_scale)
        self.last_routing: Optional[torch.Tensor] = None

    def forward(self, pet_tokens: torch.Tensor, mri_tokens: torch.Tensor, table: torch.Tensor, table_mask: torch.Tensor) -> TableFusionOutput:
        table_tokens, table_kpm, _ = self.table(table, table_mask)
        grounded_delta, attn = self.table_reads_mri(table_tokens, mri_tokens)
        grounded = table_tokens + grounded_delta
        q = self.q(mri_tokens)
        k = self.k(grounded[:, 1:])
        v = self.v(grounded[:, 1:])
        logits = torch.matmul(q, k.transpose(-1, -2)) / max(q.shape[-1] ** 0.5 * self.routing_temperature, 1e-6)
        missing = table_mask > 0.5
        logits = logits.masked_fill(missing[:, None, :], -1e4)
        all_missing = missing.all(dim=1, keepdim=True)
        routing = torch.softmax(logits, dim=-1)
        routing = torch.where(all_missing[:, :, None], torch.zeros_like(routing), routing)
        context = torch.matmul(routing, v)
        gamma, beta = self.pet_mod(context)
        self.last_routing = routing.detach().cpu()
        entropy = (-(routing.clamp_min(1e-8) * routing.clamp_min(1e-8).log()).sum(dim=-1).mean()).item()
        max_weight = routing.max(dim=-1).values.mean().item()
        top1 = routing.argmax(dim=-1)
        top1_freq = torch.bincount(top1.reshape(-1), minlength=routing.shape[-1]).float()
        top1_freq = (top1_freq / top1_freq.sum().clamp_min(1.0)).max().item()
        return TableFusionOutput(
            pet_tokens=pet_tokens,
            table_tokens=grounded,
            gamma=gamma,
            beta=beta,
            diagnostics={
                "table_mri_attn_entropy": float((-(attn.detach().mean(dim=1).clamp_min(1e-8) * attn.detach().mean(dim=1).clamp_min(1e-8).log()).sum(dim=-1).mean()).item()),
                "routing_entropy": float(entropy),
                "routing_max_weight": float(max_weight),
                "routing_top1_frequency": float(top1_freq),
                "gamma_mean": float(gamma.detach().mean().item()),
                "gamma_std": float(gamma.detach().std(unbiased=False).item()),
                "beta_mean": float(beta.detach().mean().item()),
                "beta_std": float(beta.detach().std(unbiased=False).item()),
            },
        )


class Exp032ITableFusionUNet3D(Exp032HGlobalLocalUNet3D):
    def __init__(
        self,
        table_mode: str,
        table_num_features: int,
        feature_type_ids: Sequence[int],
        table_dim: int = 64,
        table_heads: int = 4,
        table_gamma_scale: float = 0.5,
        table_beta_scale: float = 0.5,
        routing_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.table_mode = str(table_mode).lower()
        self.table_dim = int(table_dim)
        self.pet_token = PatchTokenProjector(self.channels[-1], self.table_dim)
        self.mri_token = PatchTokenProjector(self.channels[0], self.table_dim)
        self.pet_pos = nn.Parameter(torch.zeros(1, 64, self.table_dim))
        self.mri_pos = nn.Parameter(torch.zeros(1, 64, self.table_dim))
        if self.table_mode == "petbica":
            self.table_fusion = PETBiCATableFusion(self.table_dim, self.channels[-1], table_num_features, feature_type_ids, table_gamma_scale, table_beta_scale, num_heads=table_heads)
        elif self.table_mode == "dualbica":
            self.table_fusion = DualBiCATableFusion(self.table_dim, self.channels[-1], table_num_features, feature_type_ids, table_gamma_scale, table_beta_scale, num_heads=table_heads)
        elif self.table_mode == "structroute":
            self.table_fusion = StructRouteTableFusion(self.table_dim, self.channels[-1], table_num_features, feature_type_ids, table_gamma_scale, table_beta_scale, routing_temperature=routing_temperature, num_heads=table_heads)
        else:
            raise ValueError(f"Unsupported table_mode={table_mode}")
        self.last_table_stats: dict[str, float] = {}

    def load_backbone_state(self, state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        own.update({k: v for k, v in state.items() if k in own and own[k].shape == v.shape})
        self.load_state_dict(own, strict=True)

    def table_module_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if k.startswith(("pet_token.", "mri_token.", "pet_pos", "mri_pos", "table_fusion."))}

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        mri_patch: torch.Tensor,
        table: torch.Tensor,
        table_mask: torch.Tensor,
        control_enabled: bool = True,
        global_enabled: bool = True,
        global_shuffle: bool = False,
        table_enabled: bool = True,
        table_shuffle: bool = False,
        extra_missing_prob: float = 0.0,
    ) -> torch.Tensor:
        bpatch = x_t.shape[0]
        if bpatch % 64 != 0:
            raise ValueError(f"Expected flattened 64-patch batch, got {bpatch}.")
        b = bpatch // 64
        if table_shuffle:
            table = torch.roll(table, shifts=1, dims=0)
            table_mask = torch.roll(table_mask, shifts=1, dims=0)
        if extra_missing_prob > 0:
            drop = (torch.rand_like(table_mask) < float(extra_missing_prob)).float()
            table_mask = torch.maximum(table_mask, drop)
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)
        mri = self.mri_encoder(mri_patch)
        x = self.input_conv(x)
        skips = []
        global_prev = None
        levels = ["h", "m", "l"]
        x_group = None
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                x = block(x, t_emb) if isinstance(block, ResidualTimeBlock3D) else block(x)
            x = self.control(x, mri[levels[idx]], levels[idx], enabled=control_enabled)
            x_group = x.view(b, 64, x.shape[1], *x.shape[-3:])
            do_level = self.strategy == "bifusion" or levels[idx] == "l"
            if do_level:
                x_group, g = self._run_global(x_group, levels[idx], enabled=global_enabled, shuffle=global_shuffle, prev=global_prev)
                if levels[idx] == "h":
                    global_prev = self.h_to_m(g)
                elif levels[idx] == "m":
                    global_prev = self.m_to_l(g)
            if levels[idx] == "l":
                pet_tokens = self.pet_token(x_group) + self.pet_pos
                mri_high_group = mri["h"].view(b, 64, mri["h"].shape[1], *mri["h"].shape[-3:])
                mri_tokens = self.mri_token(mri_high_group) + self.mri_pos
                fused = self.table_fusion(pet_tokens, mri_tokens, table, table_mask)
                self.last_table_stats = dict(fused.diagnostics)
                if table_enabled:
                    gamma = fused.gamma[:, :, :, None, None, None].to(dtype=x_group.dtype)
                    beta = fused.beta[:, :, :, None, None, None].to(dtype=x_group.dtype)
                    x_group = (1.0 + gamma) * x_group + beta
                x = x_group.reshape(bpatch, x_group.shape[2], *x_group.shape[-3:])
            else:
                x = x_group.reshape(bpatch, x_group.shape[2], *x_group.shape[-3:])
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        for block in self.mid_blocks:
            x = block(x, t_emb) if isinstance(block, ResidualTimeBlock3D) else block(x)
        skips = list(reversed(skips[:-1]))
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, t_emb) if isinstance(block, ResidualTimeBlock3D) else block(x)
        return self.out_conv(self.out_act(self.out_norm(x)))
