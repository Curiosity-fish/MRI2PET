from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import nn

from .blocks import safe_num_heads


try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - depends on the training environment
    Mamba = None


def _scan_orders(grid_size: tuple[int, int, int]) -> torch.Tensor:
    coords = [
        (z, y, x)
        for z in range(grid_size[0])
        for y in range(grid_size[1])
        for x in range(grid_size[2])
    ]
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    orders = []
    for permutation in permutations:
        order = sorted(range(len(coords)), key=lambda i: tuple(coords[i][axis] for axis in permutation))
        orders.append(order)
    return torch.tensor(orders, dtype=torch.long)


class PatchTableInteraction(nn.Module):
    def __init__(self, token_dim: int, heads: int) -> None:
        super().__init__()
        heads = safe_num_heads(token_dim, heads)
        self.patch_norm = nn.LayerNorm(token_dim)
        self.table_norm = nn.LayerNorm(token_dim)
        self.patch_reads_table = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.table_reads_patch = nn.MultiheadAttention(token_dim, heads, batch_first=True)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        table_tokens: Optional[torch.Tensor],
        table_key_padding_mask: Optional[torch.Tensor],
        reliability: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if table_tokens is None:
            return patch_tokens, None
        table_delta = self.table_reads_patch(
            self.table_norm(table_tokens),
            self.patch_norm(patch_tokens),
            self.patch_norm(patch_tokens),
            need_weights=False,
        )[0]
        table_tokens = table_tokens + table_delta
        patch_delta = self.patch_reads_table(
            self.patch_norm(patch_tokens),
            self.table_norm(table_tokens),
            self.table_norm(table_tokens),
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )[0]
        if reliability is not None:
            patch_delta = patch_delta * reliability[:, None]
        patch_tokens = patch_tokens + patch_delta
        return patch_tokens, table_tokens


class _MambaPair(nn.Module):
    def __init__(self, dim: int, state_dim: int, conv_width: int, expand: int) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "global_mamba.backend='mamba' requires mamba-ssm. "
                "Install mamba-ssm in the training environment."
            )
        self.forward_scan = Mamba(d_model=dim, d_state=state_dim, d_conv=conv_width, expand=expand)
        self.backward_scan = Mamba(d_model=dim, d_state=state_dim, d_conv=conv_width, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward = self.forward_scan(x)
        backward = self.backward_scan(torch.flip(x, dims=(1,)))
        return 0.5 * (forward + torch.flip(backward, dims=(1,)))


class _DebugBidirectionalGRU(nn.Module):
    """Shape-test backend only; never selected silently in place of Mamba."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.gru(x)[0])


class GlobalBiMamba3D(nn.Module):
    """Multi-axis bidirectional Mamba over one summary token per PET patch."""

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        latent_channels: int,
        mri_channels: int,
        token_dim: int = 128,
        depth: int = 2,
        state_dim: int = 16,
        conv_width: int = 4,
        expand: int = 2,
        table_heads: int = 4,
        table_token_dim: int = 128,
        time_dim: int = 128,
        backend: str = "mamba",
        allow_test_backend: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.token_dim = int(token_dim)
        self.latent_proj = nn.Linear(latent_channels, token_dim)
        self.mri_proj = nn.Linear(mri_channels, token_dim)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, token_dim))
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim)
        )
        self.table_in = (
            nn.Identity()
            if int(table_token_dim) == int(token_dim)
            else nn.Linear(table_token_dim, token_dim)
        )
        self.table_out = (
            nn.Identity()
            if int(table_token_dim) == int(token_dim)
            else nn.Linear(token_dim, table_token_dim)
        )
        self.interaction_before = PatchTableInteraction(token_dim, table_heads)
        self.interaction_after = PatchTableInteraction(token_dim, table_heads)
        self.pre_norms = nn.ModuleList([nn.LayerNorm(token_dim) for _ in range(int(depth))])
        backend = str(backend).lower()
        if backend == "mamba":
            self.layers = nn.ModuleList(
                [_MambaPair(token_dim, state_dim, conv_width, expand) for _ in range(int(depth))]
            )
        elif backend == "debug_gru" and bool(allow_test_backend):
            warnings.warn("Using debug_gru instead of Mamba; this backend is for smoke tests only.")
            self.layers = nn.ModuleList([_DebugBidirectionalGRU(token_dim) for _ in range(int(depth))])
        else:
            raise ValueError(
                "backend must be 'mamba'; 'debug_gru' also requires allow_test_backend=true"
            )
        self.register_buffer("orders", _scan_orders(self.grid_size), persistent=False)
        self.out_norm = nn.LayerNorm(token_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self,
        noisy_latent_patches: torch.Tensor,
        mri_high_patches: torch.Tensor,
        time_embedding: torch.Tensor,
        table_tokens: Optional[torch.Tensor] = None,
        table_key_padding_mask: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if noisy_latent_patches.ndim != 6 or noisy_latent_patches.shape[1] != self.num_patches:
            raise ValueError(f"Expected [B,{self.num_patches},C,D,H,W] latent patches")
        latent_summary = noisy_latent_patches.mean(dim=(-3, -2, -1))
        mri_summary = mri_high_patches.mean(dim=(-3, -2, -1))
        tokens = self.latent_proj(latent_summary) + self.mri_proj(mri_summary)
        tokens = tokens + self.position + self.time_proj(time_embedding)[:, None]
        interaction_table = self.table_in(table_tokens) if table_tokens is not None else None
        tokens, interaction_table = self.interaction_before(
            tokens, interaction_table, table_key_padding_mask, reliability
        )
        for norm, layer in zip(self.pre_norms, self.layers):
            normalized = norm(tokens)
            directional = []
            for order in self.orders:
                scanned = layer(normalized.index_select(1, order))
                inverse = torch.argsort(order)
                directional.append(scanned.index_select(1, inverse))
            tokens = tokens + torch.stack(directional, dim=0).mean(dim=0)
        tokens, interaction_table = self.interaction_after(
            tokens, interaction_table, table_key_padding_mask, reliability
        )
        output_table = self.table_out(interaction_table) if interaction_table is not None else None
        return self.out_norm(tokens), output_table
