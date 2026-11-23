"""Parc98 metabolic tokens.

Builds [B, 98, C] conditioning tokens for the residual diffusion, each token
combining:
  * a learnable region identity embedding
  * a predicted region-uptake scalar embedding (from the metabolic prior)
  * a subject/table global condition
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class Parc98MetabolicTokens(nn.Module):
    def __init__(
        self,
        num_regions: int = 98,
        token_dim: int = 128,
        table_token_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        self.token_dim = int(token_dim)
        self.identity = nn.Embedding(self.num_regions, self.token_dim)
        self.uptake_embed = nn.Sequential(
            nn.Linear(1, self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.table_cond = nn.Sequential(
            nn.Linear(table_token_dim, self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.norm = nn.LayerNorm(self.token_dim)
        nn.init.trunc_normal_(self.identity.weight, std=0.02)
        nn.init.zeros_(self.table_cond[-1].weight)
        nn.init.zeros_(self.table_cond[-1].bias)

    def forward(
        self,
        region_uptake: torch.Tensor,
        table_tokens: Optional[torch.Tensor] = None,
        subject_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if region_uptake.ndim != 2 or region_uptake.shape[1] != self.num_regions:
            raise ValueError(
                f"Expected region_uptake [B,{self.num_regions}], got {tuple(region_uptake.shape)}"
            )
        batch = region_uptake.shape[0]
        device = region_uptake.device
        ids = torch.arange(self.num_regions, device=device)
        identity = self.identity(ids)[None].expand(batch, -1, -1)
        uptake = self.uptake_embed(region_uptake.unsqueeze(-1))
        if table_tokens is not None:
            summary = table_tokens.mean(dim=1)
            if subject_token is not None:
                summary = summary + subject_token
        elif subject_token is not None:
            summary = subject_token
        else:
            summary = torch.zeros((batch, self.token_dim), device=device)
        table_condition = self.table_cond(summary)[:, None, :]
        tokens = identity + uptake + table_condition
        return self.norm(tokens)  # [B, 98, token_dim]


__all__ = ["Parc98MetabolicTokens"]
