"""Components used by the MRI2PET models."""

from __future__ import annotations
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F


class MissingAwareFeatureTokenizer(nn.Module):
    """No-CLS missing-aware clinical tokenizer.

    table_mask:
        0 -> observed
        1 -> missing
    """

    def __init__(
        self,
        num_features: int,
        token_dim: int,
        feature_type_ids: Sequence[int],
        num_feature_types: int = 5,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)

        feature_type_ids = list(feature_type_ids)
        if len(feature_type_ids) != self.num_features:
            raise ValueError(
                f"feature_type_ids has {len(feature_type_ids)} entries, "
                f"expected {self.num_features}"
            )

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

        # Used only to prevent all-masked Transformer attention.
        # Image updates are reliability-gated, so all-missing still gives an
        # exact no-op on the U-Net feature pathway.
        self.null_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.norm = nn.LayerNorm(self.token_dim)

        nn.init.trunc_normal_(self.feature_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.missing_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.null_token, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        missing_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or values.shape[1] != self.num_features:
            raise ValueError(f"Expected table [B,{self.num_features}], got {tuple(values.shape)}")

        if missing_mask is None:
            missing_mask = torch.zeros_like(values)

        missing = missing_mask > 0.5
        observed = ~missing
        clean_values = torch.where(observed, values, torch.zeros_like(values)).float()

        batch = values.shape[0]
        feature_ids = torch.arange(self.num_features, device=values.device)

        tokens = (
            self.feature_embed(feature_ids)[None].expand(batch, -1, -1)
            + self.type_embed(self.feature_type_ids)[None].expand(batch, -1, -1)
            + self.value_embed(clean_values.unsqueeze(-1))
            + self.missing_embed(missing.long())
        )
        tokens = self.norm(tokens)

        key_padding_mask = missing.clone()
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            tokens = tokens.clone()
            key_padding_mask = key_padding_mask.clone()
            tokens[all_missing, 0:1] = self.null_token.to(tokens.dtype)
            key_padding_mask[all_missing, 0] = False

        reliability = observed.float().mean(dim=1, keepdim=True)
        return tokens, key_padding_mask, observed, reliability


class ClinicalTableEncoder(nn.Module):
    """Clinical feature tokens -> semantic Table tokens + subject summary."""

    def __init__(
        self,
        num_features: int,
        feature_type_ids: Sequence[int],
        token_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        num_feature_types: int = 5,
    ) -> None:
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError(f"token_dim ({token_dim}) must be divisible by heads ({heads})")

        self.tokenizer = MissingAwareFeatureTokenizer(
            num_features=num_features,
            token_dim=token_dim,
            feature_type_ids=feature_type_ids,
            num_feature_types=num_feature_types,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * int(ff_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.token_norm = nn.LayerNorm(token_dim)

        self.subject_mlp = nn.Sequential(
            nn.Linear(token_dim + 1, token_dim * 2),
            nn.SiLU(),
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        table: torch.Tensor,
        table_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        tokens, key_padding_mask, observed, reliability = self.tokenizer(table, table_mask)
        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        tokens = self.token_norm(tokens)

        weights = observed.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        subject = self.subject_mlp(torch.cat([pooled, reliability.to(pooled.dtype)], dim=1))

        return {
            "tokens": tokens,
            "key_padding_mask": key_padding_mask,
            "effective_observed": observed,
            "reliability": reliability,
            "subject_token": subject,
        }
