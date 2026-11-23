"""Components used by the MRI2PET models."""

from __future__ import annotations
import math
from typing import Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class ViTBlock(nn.Module):
    """Pre-norm Transformer block for dense bottleneck spatial tokens."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"token_dim ({dim}) must be divisible by heads ({heads})")

        hidden = int(round(dim * float(mlp_ratio)))
        if hidden <= 0:
            raise ValueError("ViT MLP hidden dimension must be > 0")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(float(dropout))

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.norm1(x)
        attn_out, _ = self.attn(
            qkv,
            qkv,
            qkv,
            need_weights=False,
        )
        x = x + self.attn_drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalViT3D(nn.Module):
    """ViT global modeling over the 5x6x5 bottleneck (150 tokens).

    No CLS token is used because Exp037C needs a dense global spatial feature
    map for the Global Pyramid and multi-scale fusion.

    Position is encoded with learnable factorized 3D embeddings:
        P(d,h,w) = P_D(d) + P_H(h) + P_W(w)
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bottleneck_shape: Tuple[int, int, int] = (5, 6, 5),
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.bottleneck_shape = tuple(int(v) for v in bottleneck_shape)

        if len(self.bottleneck_shape) != 3 or min(self.bottleneck_shape) <= 0:
            raise ValueError(f"Invalid bottleneck_shape={self.bottleneck_shape}")
        if int(depth) <= 0:
            raise ValueError("ViT depth must be > 0")
        if int(heads) <= 0:
            raise ValueError("ViT heads must be > 0")
        if self.token_dim % int(heads) != 0:
            raise ValueError(f"token_dim={self.token_dim} must be divisible by heads={heads}")

        self.in_proj = nn.Conv3d(
            self.in_channels,
            self.token_dim,
            kernel_size=1,
        )

        d, h, w = self.bottleneck_shape
        self.pos_d = nn.Parameter(torch.zeros(1, d, 1, 1, self.token_dim))
        self.pos_h = nn.Parameter(torch.zeros(1, 1, h, 1, self.token_dim))
        self.pos_w = nn.Parameter(torch.zeros(1, 1, 1, w, self.token_dim))
        nn.init.trunc_normal_(self.pos_d, std=0.02)
        nn.init.trunc_normal_(self.pos_h, std=0.02)
        nn.init.trunc_normal_(self.pos_w, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=self.token_dim,
                    heads=int(heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    attention_dropout=float(attention_dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Conv3d(
            self.token_dim,
            self.in_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")

        b, _, d, h, w = x.shape
        if (d, h, w) != self.bottleneck_shape:
            raise ValueError(
                "Unexpected bottleneck spatial shape: "
                f"got {(d,h,w)}, expected {self.bottleneck_shape}. "
                "For the current 160x192x160 image / 40x48x40 latent, "
                "the Exp037A bottleneck is expected to be 5x6x5."
            )

        volume = self.in_proj(x)  # [B,E,D,H,W]
        tokens_3d = volume.permute(0, 2, 3, 4, 1)  # [B,D,H,W,E]
        tokens_3d = tokens_3d + self.pos_d + self.pos_h + self.pos_w
        tokens = tokens_3d.reshape(b, d * h * w, self.token_dim)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.out_norm(tokens)
        volume = tokens.reshape(b, d, h, w, self.token_dim)
        volume = volume.permute(0, 4, 1, 2, 3).contiguous()
        return self.out_proj(volume)
