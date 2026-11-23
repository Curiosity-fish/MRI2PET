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


class ConditionEncoder(nn.Module):
    """Fuse time, region id and tabular global vector into one conditioning vector."""
    def __init__(self, time_dim: int, num_regions: int, table_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.region_embed = nn.Embedding(int(num_regions) + 1, time_dim)
        self.table_to_time = nn.Linear(table_dim, time_dim) if table_dim > 0 else None
        self.mlp = nn.Sequential(
            nn.Linear(time_dim * 3, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

    def forward(
        self,
        time: torch.Tensor,
        region_label: torch.Tensor,
        table_global: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if region_label is None:
            region_label = torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(region_label):
            region_label = torch.as_tensor(region_label, device=device)
        if region_label.ndim > 1:
            region_label = region_label.view(region_label.shape[0], -1)[:, 0]
        region_label = region_label.to(device=device, dtype=torch.long)
        t_emb = self.time_embed(time).to(device=device)
        r_emb = self.region_embed(region_label)
        if table_global is None or self.table_to_time is None:
            tab_emb = torch.zeros_like(t_emb)
        else:
            tab_emb = self.table_to_time(table_global.to(device=device, dtype=t_emb.dtype))
        return self.mlp(torch.cat([t_emb, r_emb, tab_emb], dim=-1))


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
    """Memory-light but expressive local block.

    It combines pointwise expansion, depthwise local convolution, depthwise dilated
    convolution and SE channel gating. No voxel-to-voxel attention is used.
    """
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


class BottleneckTableCrossAttention(nn.Module):
    """Very cheap table cross-attention at a fixed small pooled grid.

    Unlike 3D global self-attention, this never builds an N_voxel x N_voxel matrix.
    It pools the bottleneck to at most grid_size^3 image tokens and lets them attend
    to a small number of table tokens.
    """
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
        q = pooled.flatten(2).transpose(1, 2)  # B, G, C; G <= grid_size^3
        kv = self.table_proj(table_tokens.to(dtype=q.dtype))
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, c, grid_d, grid_h, grid_w)
        attn_out = F.interpolate(attn_out, size=(d, h, w), mode="trilinear", align_corners=False)
        attn_out = self.out(attn_out)
        gate = self.gate(table_global.to(dtype=q.dtype))[:, :, None, None, None].to(dtype=x.dtype)
        return x + gate * attn_out


class Focus23MGLocalDenoiser(nn.Module):
    """Exp021: Memory-efficient Mask-Guided local residual bridge denoiser.

    Inputs used:
      x_t, MRI crop, region mask crop, diffusion time, region label,
      table values and table missing mask.

    Output:
      residual R_hat with the same shape as x_t.
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
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.mri_in_channels = int(mri_in_channels)
        self.out_channels = int(out_channels)
        self.table_num_features = int(table_num_features)
        self.use_table_cross_attn = bool(use_table_cross_attn) and self.table_num_features > 0
        self.use_checkpoint = bool(use_checkpoint)

        self.table_tokenizer = (
            MissingAwareTableTokenizer(self.table_num_features, int(table_token_dim))
            if self.table_num_features > 0
            else None
        )
        self.cond_encoder = ConditionEncoder(
            time_dim=int(time_dim),
            num_regions=int(num_regions),
            table_dim=int(table_token_dim) if self.table_num_features > 0 else 0,
            cond_dim=int(cond_dim),
        )

        # input = x_t + MRI + region_mask + x_t*mask + MRI*mask
        in_ch = self.latent_dim + self.mri_in_channels + 1 + self.latent_dim + self.mri_in_channels
        chs = [int(base_channels) * int(m) for m in channel_mults]
        self.stem = nn.Conv3d(in_ch, chs[0], kernel_size=3, padding=1)

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

        self.final = nn.Sequential(
            SafeGroupNorm(chs[0], int(norm_groups)),
            nn.SiLU(),
            nn.Conv3d(chs[0], self.out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)

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
        region_mask = region_mask.to(device=x_t.device, dtype=x_t.dtype)
        mri = mri.to(device=x_t.device, dtype=x_t.dtype)

        table_tokens = table_kpm = table_global = None
        if self.table_tokenizer is not None and table is not None:
            table = table.to(device=x_t.device)
            table_mask = table_mask.to(device=x_t.device) if table_mask is not None else None
            table_tokens, table_kpm, table_global = self.table_tokenizer(table, table_mask)
            if table_tokens is not None:
                table_tokens = table_tokens.to(dtype=x_t.dtype)
            if table_global is not None:
                table_global = table_global.to(dtype=x_t.dtype)

        cond = self.cond_encoder(
            time=time,
            region_label=region_label,
            table_global=table_global,
            batch_size=x_t.shape[0],
            device=x_t.device,
        ).to(dtype=x_t.dtype)

        inp = torch.cat([x_t, mri, region_mask, x_t * region_mask, mri * region_mask], dim=1)
        h = self.stem(inp)
        skips = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = self._run_block(block, h, cond)
            skips.append(h)
            if level < len(self.downs):
                h = self.downs[level](h)

        for block in self.mid_blocks:
            h = self._run_block(block, h, cond)
        if self.table_bottleneck is not None:
            h = self.table_bottleneck(h, table_tokens, table_kpm, table_global)

        # remove bottleneck skip; decoder uses remaining high-resolution skips
        skips = skips[:-1]
        for up, blocks in zip(self.ups, self.dec_blocks):
            skip = skips.pop()
            h = up(h, skip, cond)
            for block in blocks:
                h = self._run_block(block, h, cond)

        return self.final(h)


# Backward-friendly alias
UNet = Focus23MGLocalDenoiser
