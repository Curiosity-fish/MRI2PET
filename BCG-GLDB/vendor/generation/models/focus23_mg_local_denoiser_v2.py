from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        groups = min(int(groups), channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t is None:
            raise ValueError("time/t must be provided.")
        if t.ndim > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-scale))
        emb = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MissingAwareTableTokenizer(nn.Module):
    """Turn clinical table values into tokens and a global vector.

    values: (B, F), missing_mask: (B, F), missing=1, observed=0.
    """
    def __init__(self, num_features: int, token_dim: int = 64) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        self.feature_embed = nn.Embedding(self.num_features, self.token_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(1, self.token_dim), nn.SiLU(), nn.Linear(self.token_dim, self.token_dim)
        )
        self.missing_proj = nn.Sequential(
            nn.Linear(1, self.token_dim), nn.SiLU(), nn.Linear(self.token_dim, self.token_dim)
        )
        self.missing_ratio_proj = nn.Sequential(
            nn.Linear(1, self.token_dim), nn.SiLU(), nn.Linear(self.token_dim, self.token_dim)
        )
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(
        self,
        values: Optional[torch.Tensor],
        missing_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if values is None:
            return None, None, None
        if values.ndim != 2:
            raise ValueError(f"table values must be (B,F), got {tuple(values.shape)}")
        b, f = values.shape
        if f != self.num_features:
            raise ValueError(f"Expected {self.num_features} table features, got {f}")
        values = values.float()
        if missing_mask is None:
            missing_mask = torch.zeros_like(values)
        missing_mask = missing_mask.float().to(device=values.device)
        feat_ids = torch.arange(self.num_features, device=values.device)
        feat_e = self.feature_embed(feat_ids)[None, :, :].expand(b, -1, -1)
        val_e = self.value_proj(values.unsqueeze(-1))
        miss_e = self.missing_proj(missing_mask.unsqueeze(-1))
        tokens = self.norm(feat_e + val_e + miss_e)
        key_padding_mask = missing_mask > 0.5

        valid = (~key_padding_mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_vec = (tokens * valid.unsqueeze(-1)).sum(dim=1) / denom
        miss_ratio = missing_mask.mean(dim=1, keepdim=True)
        global_vec = self.norm(global_vec + self.missing_ratio_proj(miss_ratio))
        return tokens, key_padding_mask, global_vec


class MaskShapeEncoder(nn.Module):
    """Encode the binary region mask into a compact shape/location token.

    This is intentionally light-weight. It uses only the provided crop mask and does
    not require global atlas coordinates at training time.
    """
    def __init__(self, token_dim: int = 64, hidden: int = 8) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        # 8 handcrafted stats: volume, center xyz, spread xyz, boundary ratio
        self.stat_proj = nn.Sequential(nn.Linear(8, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.cnn_proj = nn.Linear(hidden * 2, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    @staticmethod
    def _mask_stats(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        b, _, d, h, w = mask.shape
        mask_f = mask.float().clamp(0.0, 1.0)
        vol = mask_f.mean(dim=(1, 2, 3, 4), keepdim=False).view(b, 1)
        # normalized coordinates in [-1, 1]
        zz = torch.linspace(-1.0, 1.0, d, device=mask.device, dtype=mask_f.dtype).view(1, 1, d, 1, 1)
        yy = torch.linspace(-1.0, 1.0, h, device=mask.device, dtype=mask_f.dtype).view(1, 1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=mask.device, dtype=mask_f.dtype).view(1, 1, 1, 1, w)
        denom = mask_f.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(eps)
        cz = (mask_f * zz).sum(dim=(1, 2, 3, 4), keepdim=True) / denom
        cy = (mask_f * yy).sum(dim=(1, 2, 3, 4), keepdim=True) / denom
        cx = (mask_f * xx).sum(dim=(1, 2, 3, 4), keepdim=True) / denom
        sz = torch.sqrt(((mask_f * (zz - cz) ** 2).sum(dim=(1, 2, 3, 4), keepdim=True) / denom).clamp_min(eps))
        sy = torch.sqrt(((mask_f * (yy - cy) ** 2).sum(dim=(1, 2, 3, 4), keepdim=True) / denom).clamp_min(eps))
        sx = torch.sqrt(((mask_f * (xx - cx) ** 2).sum(dim=(1, 2, 3, 4), keepdim=True) / denom).clamp_min(eps))
        dilated = F.max_pool3d(mask_f, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool3d(1.0 - mask_f, kernel_size=3, stride=1, padding=1)
        boundary_ratio = (dilated - eroded).clamp(0.0, 1.0).mean(dim=(1, 2, 3, 4), keepdim=False).view(b, 1)
        stats = torch.cat([
            vol,
            cz.view(b, 1), cy.view(b, 1), cx.view(b, 1),
            sz.view(b, 1), sy.view(b, 1), sx.view(b, 1),
            boundary_ratio,
        ], dim=1)
        return stats

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().clamp(0.0, 1.0)
        cnn_token = self.cnn(mask).flatten(1)
        stats = self._mask_stats(mask).to(dtype=cnn_token.dtype)
        token = self.cnn_proj(cnn_token) + self.stat_proj(stats)
        return self.norm(token)


class RegionPrototypeMemory(nn.Module):
    """Region-specific functional prototype memory.

    Each brain region owns K learnable PET-function prototypes. A query built from
    time, region id, table, mask and image statistics retrieves a region-specific
    prototype, which is then injected into all FiLM-conditioned blocks.
    """
    def __init__(self, num_regions: int, num_prototypes: int, query_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.num_regions = int(num_regions) + 1
        self.num_prototypes = int(num_prototypes)
        self.memory_dim = int(memory_dim)
        self.query_proj = nn.Sequential(nn.Linear(query_dim, memory_dim), nn.SiLU(), nn.Linear(memory_dim, memory_dim))
        self.keys = nn.Parameter(torch.randn(self.num_regions, self.num_prototypes, self.memory_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(self.num_regions, self.num_prototypes, self.memory_dim) * 0.02)
        self.out_norm = nn.LayerNorm(memory_dim)

    def forward(self, query: torch.Tensor, region_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = query.shape[0]
        q = self.query_proj(query)
        q = F.normalize(q, dim=-1)
        region_label = region_label.view(b).to(device=query.device, dtype=torch.long)
        region_label = region_label.clamp(0, self.num_regions - 1)
        k = self.keys[region_label]
        v = self.values[region_label]
        logits = torch.einsum("bd,bkd->bk", q, F.normalize(k, dim=-1)) / math.sqrt(float(self.memory_dim))
        weights = torch.softmax(logits, dim=-1)
        prototype = torch.einsum("bk,bkd->bd", weights, v)
        return self.out_norm(prototype), weights


class RegionFunctionalConditionEncoder(nn.Module):
    """Build the main condition vector from t, region id, table, mask and prototype memory."""
    def __init__(
        self,
        time_dim: int,
        num_regions: int,
        table_dim: int,
        mask_dim: int,
        cond_dim: int,
        image_stats_dim: int = 4,
        prototype_per_region: int = 4,
        use_prototype_memory: bool = True,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.region_embed = nn.Embedding(int(num_regions) + 1, time_dim)
        self.table_to_time = nn.Linear(table_dim, time_dim) if table_dim > 0 else None
        self.mask_to_time = nn.Linear(mask_dim, time_dim)
        self.image_to_time = nn.Linear(image_stats_dim, time_dim)
        query_dim = time_dim * 5
        self.use_prototype_memory = bool(use_prototype_memory)
        self.prototype_memory = (
            RegionPrototypeMemory(
                num_regions=int(num_regions),
                num_prototypes=int(prototype_per_region),
                query_dim=query_dim,
                memory_dim=time_dim,
            )
            if self.use_prototype_memory and int(prototype_per_region) > 0
            else None
        )
        final_in = query_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(final_in, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

    def forward(
        self,
        time: torch.Tensor,
        region_label: torch.Tensor,
        table_global: Optional[torch.Tensor],
        mask_token: torch.Tensor,
        image_stats: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if region_label is None:
            region_label = torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(region_label):
            region_label = torch.as_tensor(region_label, device=device)
        if region_label.ndim > 1:
            region_label = region_label.view(region_label.shape[0], -1)[:, 0]
        region_label = region_label.to(device=device, dtype=torch.long)
        t_emb = self.time_embed(time).to(device=device)
        r_emb = self.region_embed(region_label.clamp(0, self.region_embed.num_embeddings - 1))
        if table_global is None or self.table_to_time is None:
            tab_emb = torch.zeros_like(t_emb)
        else:
            tab_emb = self.table_to_time(table_global.to(device=device, dtype=t_emb.dtype))
        mask_emb = self.mask_to_time(mask_token.to(device=device, dtype=t_emb.dtype))
        img_emb = self.image_to_time(image_stats.to(device=device, dtype=t_emb.dtype))
        query = torch.cat([t_emb, r_emb, tab_emb, mask_emb, img_emb], dim=-1)
        proto_weights = None
        if self.prototype_memory is not None:
            proto, proto_weights = self.prototype_memory(query, region_label)
        else:
            proto = torch.zeros_like(t_emb)
        cond = self.mlp(torch.cat([query, proto], dim=-1))
        return cond, proto_weights


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels * 2))
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None, None].to(dtype=x.dtype)) + shift[:, :, None, None, None].to(dtype=x.dtype)


class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class EfficientLocalBlock3D(nn.Module):
    """Memory-light local block: expansion + depthwise local/dilated conv + SE + FiLM."""
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        norm_groups: int = 8,
        expansion: float = 2.0,
        dropout: float = 0.0,
        dilation: int = 2,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = max(int(round(channels * float(expansion))), channels)
        self.norm1 = SafeGroupNorm(channels, norm_groups)
        self.film1 = FiLM(cond_dim, channels)
        self.expand = nn.Conv3d(channels, hidden, kernel_size=1)
        self.dw_local = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.dw_dilated = nn.Conv3d(
            hidden, hidden, kernel_size=3, padding=int(dilation), dilation=int(dilation), groups=hidden
        )
        self.norm2 = SafeGroupNorm(hidden, norm_groups)
        self.se = SE3D(hidden)
        self.drop = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.project = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.film1(h, cond)
        h = F.silu(h)
        h = self.expand(h)
        h = self.dw_local(h) + self.dw_dilated(h)
        h = F.silu(self.norm2(h))
        h = self.se(h)
        h = self.drop(h)
        h = self.project(h)
        return x + h


class Downsample3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SafeGroupNorm(in_ch, 8),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleAdd3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            SafeGroupNorm(in_ch, 8),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip_gate = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch), nn.Sigmoid())

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.proj(x)
        gate = self.skip_gate(cond)[:, :, None, None, None].to(dtype=x.dtype)
        return x + gate * skip


class MRIAdapterPyramid(nn.Module):
    """Light MRI-mask structural adapter, injected into each encoder level."""
    def __init__(self, mri_in_channels: int, channels: Sequence[int], norm_groups: int = 8) -> None:
        super().__init__()
        in_ch = int(mri_in_channels) + 1 + int(mri_in_channels)  # MRI + mask + MRI*mask
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0], kernel_size=3, padding=1),
            SafeGroupNorm(channels[0], norm_groups),
            nn.SiLU(),
        )
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(Downsample3D(channels[i], channels[i + 1]))

    def forward(self, mri: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        h = self.stem(torch.cat([mri, mask, mri * mask], dim=1))
        feats = [h]
        for down in self.downs:
            h = down(h)
            feats.append(h)
        return feats


class BottleneckTableCrossAttention(nn.Module):
    """Cheap bottleneck cross-attention from pooled image tokens to table tokens."""
    def __init__(
        self,
        channels: int,
        table_token_dim: int,
        heads: int = 1,
        grid_size: int = 4,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        heads = max(1, min(int(heads), self.channels))
        while heads > 1 and self.channels % heads != 0:
            heads -= 1
        self.norm = SafeGroupNorm(self.channels, norm_groups)
        self.table_proj = nn.Linear(table_token_dim, self.channels)
        self.attn = nn.MultiheadAttention(self.channels, heads, batch_first=True)
        self.out = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        self.gate = nn.Sequential(nn.Linear(table_token_dim, self.channels), nn.Sigmoid())
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x: torch.Tensor,
        table_tokens: Optional[torch.Tensor],
        table_key_padding_mask: Optional[torch.Tensor],
        table_global: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if table_tokens is None or table_global is None:
            return x
        b, c, d, h, w = x.shape
        grid_d = min(self.grid_size, d)
        grid_h = min(self.grid_size, h)
        grid_w = min(self.grid_size, w)
        pooled = F.adaptive_avg_pool3d(self.norm(x), output_size=(grid_d, grid_h, grid_w))
        q = pooled.flatten(2).transpose(1, 2)
        kv = self.table_proj(table_tokens.to(dtype=q.dtype))
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=table_key_padding_mask, need_weights=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, c, grid_d, grid_h, grid_w)
        attn_out = F.interpolate(attn_out, size=(d, h, w), mode="trilinear", align_corners=False)
        attn_out = self.out(attn_out)
        gate = self.gate(table_global.to(dtype=q.dtype))[:, :, None, None, None].to(dtype=x.dtype)
        return x + gate * attn_out


def make_mask_decomposition(mask: torch.Tensor, out_size: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return in-region, boundary band and context masks at a target feature size."""
    m = F.interpolate(mask.float(), size=out_size, mode="nearest").clamp(0.0, 1.0)
    dilated = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool3d(1.0 - m, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    context = (1.0 - dilated).clamp(0.0, 1.0)
    return m, boundary, context


def masked_feature_pool(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(device=x.device, dtype=x.dtype)
    num = (x * mask).sum(dim=(2, 3, 4))
    den = mask.sum(dim=(2, 3, 4)).clamp_min(eps)
    return num / den


class MaskGuidedStructuralAttention(nn.Module):
    """Token-level structural attention over in-region / boundary / context tokens.

    This module uses at most grid_size^3 image queries and exactly three structural
    tokens, so it is much cheaper than voxel-to-voxel self-attention.
    """
    def __init__(self, channels: int, cond_dim: int, heads: int = 1, grid_size: int = 4, norm_groups: int = 8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        heads = max(1, min(int(heads), self.channels))
        while heads > 1 and self.channels % heads != 0:
            heads -= 1
        self.norm = SafeGroupNorm(self.channels, norm_groups)
        self.attn = nn.MultiheadAttention(self.channels, heads, batch_first=True)
        self.out = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, self.channels), nn.Sigmoid())
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        xin = self.norm(x)
        minside = min(d, h, w)
        if minside < 2:
            return x
        grid_d = min(self.grid_size, d)
        grid_h = min(self.grid_size, h)
        grid_w = min(self.grid_size, w)
        m_in, m_bd, m_ctx = make_mask_decomposition(mask, out_size=(d, h, w))
        tokens = torch.stack([
            masked_feature_pool(xin, m_in),
            masked_feature_pool(xin, m_bd),
            masked_feature_pool(xin, m_ctx),
        ], dim=1)  # B, 3, C
        pooled = F.adaptive_avg_pool3d(xin, output_size=(grid_d, grid_h, grid_w))
        q = pooled.flatten(2).transpose(1, 2)  # B, G, C
        attn_out, _ = self.attn(q, tokens, tokens, need_weights=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, c, grid_d, grid_h, grid_w)
        attn_out = F.interpolate(attn_out, size=(d, h, w), mode="trilinear", align_corners=False)
        attn_out = self.out(attn_out)
        gate = self.gate(cond)[:, :, None, None, None].to(dtype=x.dtype)
        return x + gate * attn_out


class FrequencyResidualHead(nn.Module):
    """Predict residual via low-frequency and high-frequency feature branches."""
    def __init__(self, channels: int, out_channels: int, cond_dim: int, norm_groups: int = 8) -> None:
        super().__init__()
        self.low_head = nn.Sequential(
            SafeGroupNorm(channels, norm_groups),
            nn.SiLU(),
            nn.Conv3d(channels, out_channels, kernel_size=3, padding=1),
        )
        self.high_head = nn.Sequential(
            SafeGroupNorm(channels, norm_groups),
            nn.SiLU(),
            nn.Conv3d(channels, out_channels, kernel_size=3, padding=1),
        )
        self.mix_gate = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2), nn.Sigmoid())
        nn.init.zeros_(self.low_head[-1].weight)
        nn.init.zeros_(self.low_head[-1].bias)
        nn.init.zeros_(self.high_head[-1].weight)
        nn.init.zeros_(self.high_head[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # local average as a memory-free low-pass approximation
        low = F.avg_pool3d(h, kernel_size=3, stride=1, padding=1)
        high = h - low
        gate = self.mix_gate(cond).to(dtype=h.dtype)
        low_out = self.low_head(low) * gate[:, 0:1, None, None, None]
        high_out = self.high_head(high) * gate[:, 1:2, None, None, None]
        return low_out + high_out


class Focus23MGLocalDenoiser(nn.Module):
    """Exp021 second version: mask-aware region-functional frequency bridge denoiser.

    Inputs used:
      x_t, MRI crop, region mask crop, diffusion time, region label,
      table values and table missing mask.

    Innovation modules in this version:
      1) mask shape token + region functional condition token;
      2) region-specific functional prototype memory;
      3) MRI-mask gated structural adapter pyramid;
      4) mask-guided in/boundary/context structural attention at bottleneck;
      5) frequency residual head for low-frequency metabolism and high-frequency detail.
    """
    def __init__(
        self,
        latent_dim: int = 1,
        mri_in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 12,
        channel_mults: Sequence[int] = (1, 2, 4),
        blocks_per_level: int = 2,
        norm_groups: int = 8,
        dropout: float = 0.05,
        expansion: float = 2.0,
        time_dim: int = 128,
        cond_dim: int = 128,
        num_regions: int = 23,
        table_num_features: int = 0,
        table_token_dim: int = 64,
        table_cross_attn_heads: int = 1,
        table_attn_grid_size: int = 4,
        use_table_cross_attn: bool = True,
        use_checkpoint: bool = True,
        mask_token_dim: int = 64,
        prototype_per_region: int = 4,
        use_prototype_memory: bool = True,
        use_mri_adapter: bool = True,
        use_mask_guided_attention: bool = True,
        mask_attn_heads: int = 1,
        mask_attn_grid_size: int = 4,
        use_frequency_head: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.mri_in_channels = int(mri_in_channels)
        self.out_channels = int(out_channels)
        self.table_num_features = int(table_num_features)
        self.use_table_cross_attn = bool(use_table_cross_attn) and self.table_num_features > 0
        self.use_checkpoint = bool(use_checkpoint)
        self.use_mri_adapter = bool(use_mri_adapter)
        self.use_mask_guided_attention = bool(use_mask_guided_attention)
        self.use_frequency_head = bool(use_frequency_head)
        self.last_prototype_weights: Optional[torch.Tensor] = None

        self.table_tokenizer = (
            MissingAwareTableTokenizer(self.table_num_features, int(table_token_dim))
            if self.table_num_features > 0
            else None
        )
        self.mask_encoder = MaskShapeEncoder(token_dim=int(mask_token_dim), hidden=8)
        self.cond_encoder = RegionFunctionalConditionEncoder(
            time_dim=int(time_dim),
            num_regions=int(num_regions),
            table_dim=int(table_token_dim) if self.table_num_features > 0 else 0,
            mask_dim=int(mask_token_dim),
            cond_dim=int(cond_dim),
            image_stats_dim=4,
            prototype_per_region=int(prototype_per_region),
            use_prototype_memory=bool(use_prototype_memory),
        )

        # input = x_t + MRI + region_mask + x_t*mask + MRI*mask + x_t*(1-mask) + MRI*(1-mask)
        in_ch = self.latent_dim + self.mri_in_channels + 1 + self.latent_dim + self.mri_in_channels + self.latent_dim + self.mri_in_channels
        chs = [int(base_channels) * int(m) for m in channel_mults]
        self.stem = nn.Conv3d(in_ch, chs[0], kernel_size=3, padding=1)

        if self.use_mri_adapter:
            self.mri_adapter = MRIAdapterPyramid(self.mri_in_channels, chs, norm_groups=int(norm_groups))
            self.mri_gates = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), ch), nn.Sigmoid()) for ch in chs])
            # weak initial MRI injection; the model learns to increase it when useful.
            for gate in self.mri_gates:
                if isinstance(gate[-1], nn.Sigmoid) and isinstance(gate[-2], nn.Linear):
                    nn.init.zeros_(gate[-2].weight)
                    nn.init.constant_(gate[-2].bias, -2.0)
        else:
            self.mri_adapter = None
            self.mri_gates = nn.ModuleList()

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(chs):
            self.enc_blocks.append(
                nn.ModuleList([
                    EfficientLocalBlock3D(
                        ch,
                        cond_dim=int(cond_dim),
                        norm_groups=int(norm_groups),
                        expansion=float(expansion),
                        dropout=float(dropout),
                        dilation=2,
                    )
                    for _ in range(int(blocks_per_level))
                ])
            )
            if i < len(chs) - 1:
                self.downs.append(Downsample3D(ch, chs[i + 1]))

        mid_ch = chs[-1]
        self.mid_blocks = nn.ModuleList([
            EfficientLocalBlock3D(mid_ch, int(cond_dim), int(norm_groups), float(expansion), float(dropout), dilation=2),
            EfficientLocalBlock3D(mid_ch, int(cond_dim), int(norm_groups), float(expansion), float(dropout), dilation=3),
        ])
        self.mask_struct_attn = (
            MaskGuidedStructuralAttention(
                channels=mid_ch,
                cond_dim=int(cond_dim),
                heads=int(mask_attn_heads),
                grid_size=int(mask_attn_grid_size),
                norm_groups=int(norm_groups),
            )
            if self.use_mask_guided_attention
            else None
        )
        self.table_bottleneck = (
            BottleneckTableCrossAttention(
                channels=mid_ch,
                table_token_dim=int(table_token_dim),
                heads=int(table_cross_attn_heads),
                grid_size=int(table_attn_grid_size),
                norm_groups=int(norm_groups),
            )
            if self.use_table_cross_attn
            else None
        )

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(UpsampleAdd3D(chs[i + 1], chs[i], cond_dim=int(cond_dim)))
            self.dec_blocks.append(
                nn.ModuleList([
                    EfficientLocalBlock3D(
                        chs[i],
                        cond_dim=int(cond_dim),
                        norm_groups=int(norm_groups),
                        expansion=float(expansion),
                        dropout=float(dropout),
                        dilation=2,
                    )
                    for _ in range(int(blocks_per_level))
                ])
            )

        if self.use_frequency_head:
            self.final = FrequencyResidualHead(chs[0], self.out_channels, int(cond_dim), int(norm_groups))
        else:
            self.final = nn.Sequential(
                SafeGroupNorm(chs[0], int(norm_groups)),
                nn.SiLU(),
                nn.Conv3d(chs[0], self.out_channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.final[-1].weight)
            nn.init.zeros_(self.final[-1].bias)

    @staticmethod
    def _image_stats(x_t: torch.Tensor, mri: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mask = mask.float().clamp(0.0, 1.0)
        dims = (1, 2, 3, 4)
        den = mask.sum(dim=dims).clamp_min(eps)
        x_mean = (x_t.float() * mask).sum(dim=dims) / den
        m_mean = (mri.float() * mask).sum(dim=dims) / den
        x_std = torch.sqrt((((x_t.float() - x_mean[:, None, None, None, None]) ** 2) * mask).sum(dim=dims) / den + eps)
        m_std = torch.sqrt((((mri.float() - m_mean[:, None, None, None, None]) ** 2) * mask).sum(dim=dims) / den + eps)
        return torch.stack([x_mean, m_mean, x_std, m_std], dim=1)

    def _run_block(self, block: nn.Module, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(lambda _x, _cond: block(_x, _cond), x, cond, use_reentrant=False)
        return block(x, cond)

    def forward(
        self,
        x_t: Optional[torch.Tensor] = None,
        mri: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        region_label: Optional[torch.Tensor] = None,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
        # compatibility aliases
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cate_x: Optional[torch.Tensor] = None,
        conti_x: Optional[torch.Tensor] = None,
        tabular_values: Optional[torch.Tensor] = None,
        tabular_missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t is None:
            x_t = x
        if time is None:
            time = t
        if table is None:
            table = tabular_values if tabular_values is not None else conti_x
        if table_mask is None:
            table_mask = tabular_missing_mask if tabular_missing_mask is not None else cate_x
        if x_t is None or mri is None:
            raise ValueError("x_t and mri must be provided.")
        if x_t.ndim != 5 or mri.ndim != 5:
            raise ValueError(f"x_t and mri must be 5D, got {tuple(x_t.shape)} and {tuple(mri.shape)}")
        if mri.shape[2:] != x_t.shape[2:]:
            mri = F.interpolate(mri, size=x_t.shape[2:], mode="trilinear", align_corners=False)
        if region_mask is None:
            region_mask = torch.ones((x_t.shape[0], 1, *x_t.shape[2:]), device=x_t.device, dtype=x_t.dtype)
        if region_mask.shape[2:] != x_t.shape[2:]:
            region_mask = F.interpolate(region_mask.float(), size=x_t.shape[2:], mode="nearest")
        region_mask = region_mask.to(device=x_t.device, dtype=x_t.dtype).clamp(0.0, 1.0)
        mri = mri.to(device=x_t.device, dtype=x_t.dtype)
        context_mask = 1.0 - region_mask

        table_tokens = table_kpm = table_global = None
        if self.table_tokenizer is not None and table is not None:
            table = table.to(device=x_t.device)
            table_mask = table_mask.to(device=x_t.device) if table_mask is not None else None
            table_tokens, table_kpm, table_global = self.table_tokenizer(table, table_mask)
            if table_tokens is not None:
                table_tokens = table_tokens.to(dtype=x_t.dtype)
            if table_global is not None:
                table_global = table_global.to(dtype=x_t.dtype)

        mask_token = self.mask_encoder(region_mask).to(dtype=x_t.dtype)
        image_stats = self._image_stats(x_t, mri, region_mask).to(dtype=x_t.dtype)
        cond, proto_weights = self.cond_encoder(
            time=time,
            region_label=region_label,
            table_global=table_global,
            mask_token=mask_token,
            image_stats=image_stats,
            batch_size=x_t.shape[0],
            device=x_t.device,
        )
        cond = cond.to(dtype=x_t.dtype)
        self.last_prototype_weights = proto_weights.detach() if proto_weights is not None else None

        inp = torch.cat([
            x_t,
            mri,
            region_mask,
            x_t * region_mask,
            mri * region_mask,
            x_t * context_mask,
            mri * context_mask,
        ], dim=1)
        h = self.stem(inp)
        mri_feats = self.mri_adapter(mri, region_mask) if self.mri_adapter is not None else None
        skips = []
        for level, blocks in enumerate(self.enc_blocks):
            if mri_feats is not None:
                gate = self.mri_gates[level](cond)[:, :, None, None, None].to(dtype=h.dtype)
                h = h + gate * mri_feats[level].to(dtype=h.dtype)
            for block in blocks:
                h = self._run_block(block, h, cond)
            skips.append(h)
            if level < len(self.downs):
                h = self.downs[level](h)

        for block in self.mid_blocks:
            h = self._run_block(block, h, cond)
        if self.mask_struct_attn is not None:
            h = self.mask_struct_attn(h, region_mask, cond)
        if self.table_bottleneck is not None:
            h = self.table_bottleneck(h, table_tokens, table_kpm, table_global)

        skips = skips[:-1]
        for up, blocks in zip(self.ups, self.dec_blocks):
            skip = skips.pop()
            h = up(h, skip, cond)
            for block in blocks:
                h = self._run_block(block, h, cond)

        if isinstance(self.final, FrequencyResidualHead):
            return self.final(h, cond)
        return self.final(h)


# Backward-friendly alias
UNet = Focus23MGLocalDenoiser
