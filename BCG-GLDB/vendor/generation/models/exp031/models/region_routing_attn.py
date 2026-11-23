"""Patch-to-Region cross attention with anatomical routing bias.

Patch tokens (summary of noisy residual latent + MRI control per patch) attend
to the 98 Parc98 metabolic tokens.  Attention logits receive the *soft* routing
bias  log(A + eps)  (masked to -inf where A == 0), so overlapping ROIs share
mass and fully non-overlapping ROIs are excluded.  The result is a per-patch
region conditioning token used by the local denoiser.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from exp026.models.blocks import safe_num_heads
from exp031.models.anatomical_routing import routing_attention_bias


class RegionRoutingAttention3D(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        mri_channels: int,
        token_dim: int = 128,
        num_regions: int = 98,
        num_heads: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_regions = int(num_regions)
        self.num_heads = safe_num_heads(self.token_dim, int(num_heads))
        self.head_dim = self.token_dim // self.num_heads
        self.eps = float(eps)
        self.patch_proj = nn.Linear(int(latent_channels) + int(mri_channels), self.token_dim)
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        # Zero-init the output projection so the routing branch starts as a
        # no-op and never disturbs a freshly initialised diffusion.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        noisy_latent_patches: torch.Tensor,
        mri_high_patches: torch.Tensor,
        roi_tokens: torch.Tensor,
        routing: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-patch region tokens [B, P, token_dim]."""
        if noisy_latent_patches.ndim != 6 or mri_high_patches.ndim != 6:
            raise ValueError("Expected [B,P,C,D,H,W] latent and MRI patch tensors")
        if noisy_latent_patches.shape[1] != mri_high_patches.shape[1]:
            raise ValueError("Latent and MRI patch counts must match")
        batch, patches = noisy_latent_patches.shape[:2]
        latent_summary = noisy_latent_patches.mean(dim=(-3, -2, -1))
        mri_summary = mri_high_patches.mean(dim=(-3, -2, -1))
        patch_tokens = self.patch_proj(torch.cat([latent_summary, mri_summary], dim=-1))
        query = self.q_proj(patch_tokens)  # [B,P,D]
        key = self.k_proj(roi_tokens)      # [B,R,D]
        value = self.v_proj(roi_tokens)    # [B,R,D]

        batch_heads = batch * self.num_heads
        query = (
            query.reshape(batch, patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_heads, patches, self.head_dim)
        )
        key = (
            key.reshape(batch, self.num_regions, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_heads, self.num_regions, self.head_dim)
        )
        value = (
            value.reshape(batch, self.num_regions, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_heads, self.num_regions, self.head_dim)
        )
        scores = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        patch_indices = (
            torch.arange(patches, device=routing.device)
            .view(1, -1)
            .expand(batch, -1)
            .contiguous()
        )
        bias = routing_attention_bias(
            routing, patch_indices, eps=self.eps
        )  # [B,P,R]
        # Replicate the (identical) routing bias across attention heads.
        bias = (
            bias.unsqueeze(1)
            .expand(batch, self.num_heads, patches, self.num_regions)
            .reshape(batch_heads, patches, self.num_regions)
            .contiguous()
        )
        scores = scores + bias
        attention = torch.softmax(scores, dim=-1)
        output = torch.bmm(attention, value)  # [B*H, P, head_dim]
        output = (
            output.reshape(batch, self.num_heads, patches, self.head_dim)
            .transpose(1, 2)
            .reshape(batch, patches, self.token_dim)
        )
        return self.out_proj(output)  # [B, P, token_dim]


__all__ = ["RegionRoutingAttention3D"]
