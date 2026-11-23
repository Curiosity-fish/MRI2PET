from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .blocks import SafeGroupNorm, safe_num_heads


class MissingAwareTableTokenizer(nn.Module):
    def __init__(
        self,
        num_features: int,
        token_dim: int,
        feature_type_ids: Optional[Sequence[int]] = None,
        num_feature_types: int = 1,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")
        feature_type_ids = list(feature_type_ids or [0] * self.num_features)
        if len(feature_type_ids) != self.num_features:
            raise ValueError("feature_type_ids must match num_features")
        self.register_buffer(
            "feature_type_ids",
            torch.tensor(feature_type_ids, dtype=torch.long),
            persistent=False,
        )
        self.feature_embed = nn.Embedding(self.num_features, self.token_dim)
        self.type_embed = nn.Embedding(max(1, int(num_feature_types)), self.token_dim)
        self.value_embed = nn.Sequential(
            nn.Linear(1, self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.missing_embed = nn.Embedding(2, self.token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.norm = nn.LayerNorm(self.token_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        missing_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or values.shape[1] != self.num_features:
            raise ValueError(
                f"Expected table [B,{self.num_features}], got {tuple(values.shape)}"
            )
        if missing_mask is None:
            missing_mask = torch.zeros_like(values)
        missing = missing_mask.gt(0.5)
        batch = values.shape[0]
        ids = torch.arange(self.num_features, device=values.device)
        features = self.feature_embed(ids)[None].expand(batch, -1, -1)
        types = self.type_embed(self.feature_type_ids)[None].expand(batch, -1, -1)
        tokens = features + types + self.value_embed(values.float().unsqueeze(-1))
        tokens = tokens + self.missing_embed(missing.long())
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = self.norm(torch.cat([cls, tokens], dim=1))
        key_padding_mask = torch.cat(
            [torch.zeros((batch, 1), dtype=torch.bool, device=values.device), missing], dim=1
        )
        reliability = (~missing).float().mean(dim=1, keepdim=True)
        return tokens, key_padding_mask, reliability


class MRIPatchTokenizer3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        num_patches: int,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_patches)
        self.proj = nn.Linear(in_channels, token_dim)
        self.position = nn.Parameter(torch.zeros(1, self.num_tokens, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 6 or feature.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected MRI patches [B,{self.num_tokens},C,D,H,W], got {tuple(feature.shape)}"
            )
        tokens = feature.mean(dim=(-3, -2, -1))
        return self.norm(self.proj(tokens) + self.position[:, : tokens.shape[1]])


class BidirectionalTokenBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int) -> None:
        super().__init__()
        heads = safe_num_heads(token_dim, num_heads)
        self.table_norm = nn.LayerNorm(token_dim)
        self.mri_norm = nn.LayerNorm(token_dim)
        self.table_reads_mri = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.mri_reads_table = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.table_ff_norm = nn.LayerNorm(token_dim)
        self.mri_ff_norm = nn.LayerNorm(token_dim)
        self.table_ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4), nn.GELU(), nn.Linear(token_dim * 4, token_dim)
        )
        self.mri_ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4), nn.GELU(), nn.Linear(token_dim * 4, token_dim)
        )

    def forward(
        self,
        table_tokens: torch.Tensor,
        table_key_padding_mask: torch.Tensor,
        mri_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table_delta = self.table_reads_mri(
            self.table_norm(table_tokens),
            self.mri_norm(mri_tokens),
            self.mri_norm(mri_tokens),
            need_weights=False,
        )[0]
        table_tokens = table_tokens + table_delta
        table_tokens = table_tokens + self.table_ff(self.table_ff_norm(table_tokens))
        mri_delta = self.mri_reads_table(
            self.mri_norm(mri_tokens),
            self.table_norm(table_tokens),
            self.table_norm(table_tokens),
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )[0]
        mri_tokens = mri_tokens + mri_delta
        mri_tokens = mri_tokens + self.mri_ff(self.mri_ff_norm(mri_tokens))
        return table_tokens, mri_tokens


class JointTableMRIPrior3D(nn.Module):
    def __init__(
        self,
        num_features: int,
        mri_channels: int,
        latent_channels: int,
        token_dim: int = 128,
        num_patches: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        feature_type_ids: Optional[Sequence[int]] = None,
        num_feature_types: int = 1,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        self.table_tokenizer = MissingAwareTableTokenizer(
            num_features,
            token_dim,
            feature_type_ids=feature_type_ids,
            num_feature_types=num_feature_types,
        )
        self.mri_tokenizer = MRIPatchTokenizer3D(mri_channels, token_dim, self.num_patches)
        self.blocks = nn.ModuleList(
            [BidirectionalTokenBlock(token_dim, num_heads) for _ in range(int(depth))]
        )
        self.subject = nn.Sequential(
            nn.Linear(token_dim * 2 + 1, token_dim * 2),
            nn.SiLU(),
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
        )
        output = nn.Conv3d(mri_channels, latent_channels, kernel_size=1)
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(output.bias)
        self.patch_prior_residual = nn.Sequential(
            nn.Conv3d(mri_channels, mri_channels, kernel_size=3, padding=1),
            SafeGroupNorm(mri_channels, 8),
            nn.SiLU(),
            output,
        )
        self.patch_condition = nn.Linear(token_dim, mri_channels)
        self.subject_condition = nn.Linear(token_dim, mri_channels)

    def forward(
        self,
        mri_high_patches: torch.Tensor,
        values: torch.Tensor,
        missing_mask: Optional[torch.Tensor],
        base_prior_patches: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        table_tokens, table_kpm, reliability = self.table_tokenizer(values, missing_mask)
        mri_tokens = self.mri_tokenizer(mri_high_patches)
        for block in self.blocks:
            table_tokens, mri_tokens = block(table_tokens, table_kpm, mri_tokens)
        subject = self.subject(
            torch.cat([table_tokens[:, 0], mri_tokens.mean(dim=1), reliability], dim=1)
        )
        batch, patches, channels = mri_high_patches.shape[:3]
        condition = self.patch_condition(mri_tokens)
        condition = condition + self.subject_condition(subject)[:, None]
        conditioned_mri = mri_high_patches + condition[..., None, None, None]
        flat = conditioned_mri.reshape(batch * patches, channels, *conditioned_mri.shape[-3:])
        residual = self.patch_prior_residual(flat).reshape(
            batch, patches, -1, *conditioned_mri.shape[-3:]
        )
        residual = reliability[:, None, :, None, None, None] * residual
        prior = residual if base_prior_patches is None else base_prior_patches + residual
        return {
            "table_tokens": table_tokens,
            "table_key_padding_mask": table_kpm,
            "mri_patch_tokens": mri_tokens,
            "subject_token": subject,
            "reliability": reliability,
            "prior_patches": prior,
        }
