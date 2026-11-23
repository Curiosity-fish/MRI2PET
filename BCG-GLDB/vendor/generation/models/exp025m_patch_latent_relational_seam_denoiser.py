from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.exp025l_patch_latent_control_table_denoiser import (
    Exp025LPatchLatentControlTableDenoiser3D,
)


def _safe_heads(embed_dim: int, num_heads: int) -> int:
    heads = max(1, min(int(num_heads), int(embed_dim)))
    while heads > 1 and int(embed_dim) % heads != 0:
        heads -= 1
    return heads


class NeighborhoodSummaryPatchMixer(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        cond_dim: int,
        token_dim: int,
        num_heads: int,
        subject_dim: int,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(int(feat_channels), int(token_dim))
        self.cond_proj = nn.Linear(int(cond_dim), int(token_dim))
        self.subject_proj = nn.Linear(int(subject_dim), int(token_dim))
        self.norm1 = nn.LayerNorm(int(token_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(token_dim),
            num_heads=_safe_heads(int(token_dim), int(num_heads)),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(int(token_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(token_dim), int(token_dim) * 4),
            nn.GELU(),
            nn.Linear(int(token_dim) * 4, int(token_dim)),
        )
        self.delta = nn.Linear(int(token_dim), int(feat_channels))
        self.gate = nn.Sequential(
            nn.Linear(int(token_dim), int(feat_channels)),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_mask(patch_coords: torch.Tensor) -> torch.Tensor:
        patch_coords = patch_coords.long()
        patches = int(patch_coords.shape[0])
        diff = (patch_coords[:, None, :] - patch_coords[None, :, :]).abs()
        local_neighborhood = diff.amax(dim=-1) <= 1
        attn_mask = torch.ones((patches + 1, patches + 1), dtype=torch.bool, device=patch_coords.device)
        attn_mask[0, :] = False
        attn_mask[:, 0] = False
        attn_mask[1:, 1:] = ~local_neighborhood
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: torch.Tensor,
        patch_coords: torch.Tensor,
        subject_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(x, 1).flatten(1)
        tokens = self.feat_proj(pooled) + self.cond_proj(cond_vec)
        if subject_token is not None:
            tokens = tokens + self.subject_proj(subject_token)
        tokens = tokens.unsqueeze(0)
        summary = tokens.mean(dim=1, keepdim=True)
        tokens = torch.cat([summary, tokens], dim=1)
        attn_mask = self._build_mask(patch_coords)
        tokens = tokens + self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), attn_mask=attn_mask, need_weights=False)[0]
        tokens = tokens + self.ff(self.norm2(tokens))
        patch_tokens = tokens[:, 1:, :].squeeze(0)
        gate = self.gate(patch_tokens)[:, :, None, None, None].to(dtype=x.dtype)
        delta = self.delta(patch_tokens)[:, :, None, None, None].to(dtype=x.dtype)
        return x + gate * delta


class Exp025MPatchLatentRelationalSeamDenoiser3D(Exp025LPatchLatentControlTableDenoiser3D):
    def __init__(
        self,
        *args,
        patch_relation_token_dim: int = 128,
        patch_relation_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        c10, c5 = [int(v) for v in kwargs.get("local_channels", (32, 64))]
        time_dim = int(kwargs.get("time_dim", 256))
        subject_dim = int(kwargs.get("table_token_dim", 128))
        token_dim = int(patch_relation_token_dim)
        heads = int(patch_relation_heads)
        self.patch_mixer10e = NeighborhoodSummaryPatchMixer(c10, time_dim, token_dim, heads, subject_dim)
        self.patch_mixer5 = NeighborhoodSummaryPatchMixer(c5, time_dim, token_dim, heads, subject_dim)
        self.patch_mixer10d = NeighborhoodSummaryPatchMixer(c10, time_dim, token_dim, heads, subject_dim)

    def denoise_patches(
        self,
        x_t_patch: torch.Tensor,
        cond_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        time: torch.Tensor,
        global_context: dict[str, torch.Tensor],
        table_context: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        cond_vec = self._global_condition(time, global_context["pooled"], patch_coords)
        local_control = self.local_control(cond_patch)
        global10 = self._crop_feature(global_context["40"], patch_coords)
        global5 = self._crop_feature(global_context["20"], patch_coords)

        prior_patch = None
        table_tokens = None
        table_kpm = None
        subject = None
        reliability = None
        time_fraction = None
        if table_context is not None:
            prior_patch = self._crop_feature(table_context["prior_full"], patch_coords)
            table_tokens = self._expand_rows(table_context["table_tokens"], x_t_patch.shape[0])
            table_kpm = self._expand_rows(table_context["table_kpm"], x_t_patch.shape[0])
            subject = self._expand_rows(table_context["subject"], x_t_patch.shape[0])
            reliability = self._expand_rows(table_context["reliability"], x_t_patch.shape[0])
            time_fraction = (time.float() / float(self.max_timesteps - 1)).clamp(0.0, 1.0)[:, None]
            time_fraction = self._expand_rows(time_fraction, x_t_patch.shape[0])

        x = self.local_stem(x_t_patch)
        x = self.enc10a(x, cond_vec)
        x = self.enc10b(x, cond_vec)
        if prior_patch is not None and self.prompt10e is not None:
            x = self.prompt10e(x, prior_patch, subject, cond_vec, reliability)
        if table_tokens is not None and self.fusion10e is not None:
            x, table_tokens = self.fusion10e(x, table_tokens, table_kpm, subject, cond_vec, reliability, time_fraction)
        x = self._inject(x, local_control["10"], self.gate10e, self.inject10e, cond_vec)
        x = self.fuse_global10e(x, global10)
        x = self.patch_mixer10e(x, cond_vec, patch_coords, subject)
        skip10 = x

        x = self.down5(x)
        x = self.mid5a(x, cond_vec)
        if prior_patch is not None and self.prompt5 is not None:
            x = self.prompt5(x, prior_patch, subject, cond_vec, reliability)
        if table_tokens is not None and self.fusion5 is not None:
            x, table_tokens = self.fusion5(x, table_tokens, table_kpm, subject, cond_vec, reliability, time_fraction)
        x = self._inject(x, local_control["5"], self.gate5, self.inject5, cond_vec)
        x = self.fuse_global5(x, global5)
        x = self.patch_mixer5(x, cond_vec, patch_coords, subject)
        x = self.mid5b(x, cond_vec)

        x = self.up10(x, skip10.shape[-3:])
        x = self.dec10a(torch.cat([x, skip10], dim=1), cond_vec)
        x = self.dec10b(x, cond_vec)
        if prior_patch is not None and self.prompt10d is not None:
            x = self.prompt10d(x, prior_patch, subject, cond_vec, reliability)
        if table_tokens is not None and self.fusion10d is not None:
            x, _ = self.fusion10d(x, table_tokens, table_kpm, subject, cond_vec, reliability, time_fraction)
        x = self._inject(x, local_control["10"], self.gate10d, self.inject10d, cond_vec)
        x = self.fuse_global10d(x, global10)
        x = self.patch_mixer10d(x, cond_vec, patch_coords, subject)
        return self.out(x)
