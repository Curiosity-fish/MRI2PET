from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v)
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(attn, v)


def _safe_groups(channels: int, groups: int) -> int:
    channels = int(channels)
    groups = min(int(groups), channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _zero_last_linear(module: nn.Sequential) -> None:
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_safe_groups(channels, groups), int(channels), eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        emb = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


@dataclass
class MambaConfig:
    d_model: int
    d_state: int = 32
    expand_factor: int = 4
    d_conv: int = 4
    dt_rank: int | str = "auto"

    def __post_init__(self) -> None:
        self.d_inner = int(self.d_model) * int(self.expand_factor)
        if self.dt_rank == "auto":
            self.dt_rank = max(1, math.ceil(int(self.d_model) / 16))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MambaMixer(nn.Module):
    """Reference selective SSM implementation.

    This implementation favors clarity and compatibility. It is not a fused mamba_ssm kernel.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        c = config
        self.config = c
        self.in_proj = nn.Linear(c.d_model, 2 * c.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            c.d_inner,
            c.d_inner,
            kernel_size=c.d_conv,
            groups=c.d_inner,
            padding=c.d_conv - 1,
        )
        self.x_proj = nn.Linear(c.d_inner, int(c.dt_rank) + 2 * c.d_state, bias=False)
        self.dt_proj = nn.Linear(int(c.dt_rank), c.d_inner, bias=True)
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=False)
        a = torch.arange(1, c.d_state + 1, dtype=torch.float32).repeat(c.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(c.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_branch = F.silu(x_branch)
        y = self._ssm(x_branch)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        x = x.float()
        delta_bc = self.x_proj(x).float()
        delta, b_term, c_term = torch.split(
            delta_bc,
            [int(self.config.dt_rank), self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta).float())
        a_term = -torch.exp(self.A_log.float())
        d_term = self.D.float()
        batch, length, d_inner = x.shape
        state = torch.zeros(
            batch,
            d_inner,
            self.config.d_state,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []
        for index in range(length):
            dt = delta[:, index]
            dt_a = torch.exp(dt.unsqueeze(-1) * a_term.unsqueeze(0))
            dt_b = dt.unsqueeze(-1) * b_term[:, index].unsqueeze(1)
            state = dt_a * state + dt_b * x[:, index].unsqueeze(-1)
            y = (state * c_term[:, index].unsqueeze(1)).sum(dim=-1)
            y = y + d_term.unsqueeze(0) * x[:, index]
            outputs.append(y)
        return torch.stack(outputs, dim=1).to(dtype=output_dtype)


class ResidualMamba(nn.Module):
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, use_fused: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        if use_fused:
            from mamba_ssm import Mamba
            self.mixer = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mixer = MambaMixer(
                MambaConfig(dim, d_state=d_state, d_conv=d_conv, expand_factor=expand)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class TableMixerBlock(nn.Module):
    """MLP-Mixer block over table-variable tokens."""

    def __init__(self, num_tokens: int, dim: int) -> None:
        super().__init__()
        token_hidden = max(2 * int(num_tokens), 8)
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, num_tokens),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_mixed = self.token_norm(x).transpose(1, 2)
        token_mixed = self.token_mlp(token_mixed).transpose(1, 2)
        x = x + token_mixed
        return x + self.channel_mlp(self.channel_norm(x))


class TableTokenMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class TableEncoder(nn.Module):
    """Missing-aware table encoder.

    table_mask convention: 1 means missing, 0 means observed.
    """

    def __init__(
        self,
        num_features: int,
        table_dim: int,
        depth: int,
        dropout: float,
        encoder_type: str = "mixer",
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.table_dim = int(table_dim)
        self.encoder_type = str(encoder_type).lower()
        self.feature_embed = nn.Embedding(max(self.num_features, 1), self.table_dim)
        self.value_proj = nn.Linear(1, self.table_dim)
        self.missing_embed = nn.Embedding(2, self.table_dim)

        blocks: list[nn.Module] = []
        if self.num_features > 0:
            for _ in range(int(depth)):
                if self.encoder_type == "mixer":
                    blocks.append(TableMixerBlock(self.num_features, self.table_dim))
                elif self.encoder_type in {"token_mlp", "mlp"}:
                    blocks.append(TableTokenMLPBlock(self.table_dim))
                else:
                    raise ValueError(
                        f"Unsupported table encoder_type={encoder_type!r}. "
                        "Supported values are 'mixer' and 'token_mlp'."
                    )
        self.blocks = nn.ModuleList(blocks)
        self.token_norm = nn.LayerNorm(self.table_dim)
        self.token_gate = nn.Sequential(
            nn.Linear(self.table_dim, max(self.table_dim // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(self.table_dim // 2, 16), 1),
        )
        self.case_head = nn.Sequential(
            nn.Linear(self.table_dim + 1, self.table_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.table_dim, self.table_dim),
        )
        self.case_reliability = nn.Sequential(
            nn.Linear(self.table_dim + 1, max(self.table_dim // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(self.table_dim // 2, 16), 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        table: torch.Tensor,
        table_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_features <= 0:
            batch = table.shape[0]
            return (
                table.new_zeros((batch, self.table_dim)),
                table.new_zeros((batch, 1)),
            )
        if table.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} table features, got {table.shape[1]}."
            )
        missing = (table_mask > 0.5).long()
        valid = (1 - missing).float()
        safe_values = table.float() * valid
        feature_ids = torch.arange(self.num_features, device=table.device, dtype=torch.long)
        x = self.value_proj(safe_values.unsqueeze(-1))
        x = x + self.feature_embed(feature_ids)[None, :, :]
        x = x + self.missing_embed(missing)
        for block in self.blocks:
            x = block(x)
        x = self.token_norm(x)

        token_weight = torch.sigmoid(self.token_gate(x)) * valid.unsqueeze(-1)
        pooled = (x * token_weight).sum(dim=1) / token_weight.sum(dim=1).clamp_min(1e-6)
        quality = valid.mean(dim=1, keepdim=True)
        pooled = self.case_head(torch.cat([pooled, quality], dim=1))
        case_reliability = quality * self.case_reliability(
            torch.cat([pooled, quality], dim=1)
        )
        return pooled, case_reliability


class ReliabilityProjector(nn.Module):
    """Produces strict missing-aware, optionally position-aware channel gates."""

    def __init__(
        self,
        table_dim: int,
        num_table_features: int,
        out_dims: dict[str, int],
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_table_features = int(num_table_features)
        self.extra_dim = int(extra_dim)
        input_dim = int(table_dim) + 1 + self.num_table_features + self.extra_dim
        self.heads = nn.ModuleDict()
        for key, out_dim in out_dims.items():
            hidden = max(int(out_dim) // 2, 16)
            self.heads[key] = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, int(out_dim)),
                nn.Sigmoid(),
            )

    def forward(
        self,
        table_emb: torch.Tensor,
        case_reliability: torch.Tensor,
        table_mask: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        parts = [table_emb, case_reliability]
        if self.num_table_features > 0:
            parts.append(table_mask.float())
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError("extra input is required for position-aware reliability.")
            parts.append(extra)
        x = torch.cat(parts, dim=1)
        return {
            key: case_reliability * head(x)
            for key, head in self.heads.items()
        }


class PositionEncoder(nn.Module):
    def __init__(self, dim: int, grid_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        gz, gy, gx = self.grid_size
        self.ez = nn.Embedding(gz, dim)
        self.ey = nn.Embedding(gy, dim)
        self.ex = nn.Embedding(gx, dim)
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.long()
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"patch_coords must have shape [N,3], got {tuple(coords.shape)}")
        gz, gy, gx = self.grid_size
        if (
            (coords[:, 0] < 0).any()
            or (coords[:, 0] >= gz).any()
            or (coords[:, 1] < 0).any()
            or (coords[:, 1] >= gy).any()
            or (coords[:, 2] < 0).any()
            or (coords[:, 2] >= gx).any()
        ):
            raise ValueError("patch_coords contains an out-of-range coordinate.")
        cz = (coords[:, 0].float() + 0.5) / float(gz)
        cy = (coords[:, 1].float() + 0.5) / float(gy)
        cx = (coords[:, 2].float() + 0.5) / float(gx)
        continuous = torch.stack([cz, cy, cx], dim=1)
        return (
            self.ez(coords[:, 0])
            + self.ey(coords[:, 1])
            + self.ex(coords[:, 2])
            + self.mlp(continuous)
        )


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    dz = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    return torch.sqrt((dx.square() + dy.square() + dz.square()).clamp_min(1e-8))


def _log_approx(x: torch.Tensor) -> torch.Tensor:
    return x - F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)


def _build_mri_structure_input(
    mri: torch.Tensor,
    use_gradient: bool,
    use_log: bool,
) -> torch.Tensor:
    parts = [mri]
    if use_gradient:
        parts.append(_grad_mag(mri))
    if use_log:
        parts.append(_log_approx(mri))
    return torch.cat(parts, dim=1)


class ResidualConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, norm_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = SafeGroupNorm(out_channels, norm_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if int(in_channels) != int(out_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class DualGateFusion3D(nn.Module):
    def __init__(self, local_channels: int, guide_channels: int, norm_groups: int) -> None:
        super().__init__()
        self.guide_proj = nn.Conv3d(guide_channels, local_channels, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.Linear(local_channels * 2, max(local_channels // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(local_channels // 2, 16), local_channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(local_channels * 2, local_channels, kernel_size=3, padding=1),
            SafeGroupNorm(local_channels, norm_groups),
            nn.SiLU(),
            nn.Conv3d(local_channels, local_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.full((1,), 1e-2))

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        guide = self.guide_proj(guide)
        pooled = torch.cat(
            [x.mean(dim=(2, 3, 4)), guide.mean(dim=(2, 3, 4))],
            dim=1,
        )
        channel_gate = self.channel_gate(pooled)[:, :, None, None, None].to(dtype=x.dtype)
        spatial_gate = self.spatial_gate(torch.cat([x, guide], dim=1))
        return x + self.alpha.to(dtype=x.dtype) * channel_gate * spatial_gate * guide


class ConditionalResBlock3D(nn.Module):
    """Time/position modulation plus strictly reliability-gated table modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_condition_dim: int,
        table_condition_dim: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = SafeGroupNorm(in_channels, norm_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = SafeGroupNorm(out_channels, norm_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if int(in_channels) != int(out_channels)
            else nn.Identity()
        )
        self.base_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(base_condition_dim, out_channels * 3),
        )
        self.table_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(table_condition_dim, out_channels * 3),
        )
        _zero_last_linear(self.table_proj)

    def forward(
        self,
        x: torch.Tensor,
        base_condition: torch.Tensor,
        table_condition: torch.Tensor,
        table_reliability: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma_base, beta_base, alpha_base = self.base_proj(base_condition).chunk(3, dim=1)
        gamma_table, beta_table, alpha_table = self.table_proj(table_condition).chunk(3, dim=1)
        reliability = table_reliability.to(dtype=h.dtype)
        gamma = gamma_base + reliability * gamma_table
        beta = beta_base + reliability * beta_table
        alpha = torch.sigmoid(alpha_base + reliability * alpha_table)
        h = self.norm2(h)
        h = h * (1.0 + gamma[:, :, None, None, None].to(dtype=h.dtype))
        h = h + beta[:, :, None, None, None].to(dtype=h.dtype)
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + alpha[:, :, None, None, None].to(dtype=h.dtype) * h


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return self.conv(x)


class MRIStructureEncoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        norm_groups: int,
        dropout: float,
        blocks_per_level: int,
        use_gradient: bool,
        use_log: bool,
    ) -> None:
        super().__init__()
        self.use_gradient = bool(use_gradient)
        self.use_log = bool(use_log)
        input_channels = 1 + int(self.use_gradient) + int(self.use_log)
        c40, c20, c10, c5 = [int(value) for value in channels]
        self.stem = nn.Conv3d(input_channels, c40, kernel_size=3, padding=1)
        self.level40 = nn.ModuleList(
            [ResidualConvBlock3D(c40, c40, norm_groups, dropout) for _ in range(blocks_per_level)]
        )
        self.down20 = Downsample3D(c40, c20)
        self.level20 = nn.ModuleList(
            [ResidualConvBlock3D(c20, c20, norm_groups, dropout) for _ in range(blocks_per_level)]
        )
        self.down10 = Downsample3D(c20, c10)
        self.level10 = nn.ModuleList(
            [ResidualConvBlock3D(c10, c10, norm_groups, dropout) for _ in range(blocks_per_level)]
        )
        self.down5 = Downsample3D(c10, c5)
        self.level5 = nn.ModuleList(
            [ResidualConvBlock3D(c5, c5, norm_groups, dropout) for _ in range(blocks_per_level)]
        )

    def forward(self, mri_patch: torch.Tensor) -> dict[str, torch.Tensor]:
        x = _build_mri_structure_input(mri_patch, self.use_gradient, self.use_log)
        x = self.stem(x)
        for block in self.level40:
            x = block(x)
        feature40 = x
        x = self.down20(x)
        for block in self.level20:
            x = block(x)
        feature20 = x
        x = self.down10(x)
        for block in self.level10:
            x = block(x)
        feature10 = x
        x = self.down5(x)
        for block in self.level5:
            x = block(x)
        feature5 = x
        return {
            "40": feature40,
            "20": feature20,
            "10": feature10,
            "5": feature5,
        }


class GlobalStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c0, c80, c40 = [int(value) for value in channels]
        self.stem = nn.Conv3d(in_channels, c0, kernel_size=3, padding=1)
        self.down80 = nn.Conv3d(c0, c80, kernel_size=3, stride=2, padding=1)
        self.block80 = ResidualConvBlock3D(c80, c80, norm_groups, dropout)
        self.down40 = nn.Conv3d(c80, c40, kernel_size=3, stride=2, padding=1)
        self.block40 = ResidualConvBlock3D(c40, c40, norm_groups, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.stem(x))
        feature80 = self.block80(self.down80(x))
        feature40 = self.block40(self.down40(feature80))
        return feature80, feature40


class AxialBiMambaBlock(nn.Module):
    """Memory-controlled axial Mamba block.

    The spatial feature map remains unchanged. Memory is controlled by splitting
    the independent axial lines into smaller batches and optionally applying
    activation checkpointing to every Mamba line batch.
    """

    _AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        bidirectional: bool = True,
        active_axes: tuple[str, ...] = ("x", "y", "z"),
        line_chunk_size: int = 0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.base_proj = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, dim * 3))
        self.table_proj = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, dim * 3))
        _zero_last_linear(self.table_proj)

        axes = tuple(str(axis).lower() for axis in active_axes)
        if not axes or any(axis not in self._AXIS_TO_INDEX for axis in axes):
            raise ValueError(f"active_axes must be a non-empty subset of x/y/z, got {active_axes}.")
        if len(set(axes)) != len(axes):
            raise ValueError(f"active_axes contains duplicates: {active_axes}.")
        self.active_axes = axes
        self.bidirectional = bool(bidirectional)
        self.line_chunk_size = max(int(line_chunk_size), 0)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.x_forward = ResidualMamba(dim, d_state, d_conv, expand)
        self.y_forward = ResidualMamba(dim, d_state, d_conv, expand)
        self.z_forward = ResidualMamba(dim, d_state, d_conv, expand)
        self.x_backward = ResidualMamba(dim, d_state, d_conv, expand) if self.bidirectional else None
        self.y_backward = ResidualMamba(dim, d_state, d_conv, expand) if self.bidirectional else None
        self.z_backward = ResidualMamba(dim, d_state, d_conv, expand) if self.bidirectional else None

        self.axis_weight = nn.Parameter(torch.ones(3))
        self.alpha = nn.Parameter(torch.full((1,), 1e-2))
        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1)

    def _apply_mamba(self, module: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and sequence.requires_grad:
            return checkpoint(module, sequence, use_reentrant=True)
        return module(sequence)

    def _run_axis(
        self,
        x: torch.Tensor,
        axis: int,
        forward_mamba: nn.Module,
        backward_mamba: Optional[nn.Module],
    ) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        if axis == 0:
            sequence = x.permute(0, 3, 4, 2, 1).reshape(batch * height * width, depth, channels)
            restore = lambda value: value.reshape(batch, height, width, depth, channels).permute(0, 4, 3, 1, 2)
        elif axis == 1:
            sequence = x.permute(0, 2, 4, 3, 1).reshape(batch * depth * width, height, channels)
            restore = lambda value: value.reshape(batch, depth, width, height, channels).permute(0, 4, 1, 3, 2)
        else:
            sequence = x.permute(0, 2, 3, 4, 1).reshape(batch * depth * height, width, channels)
            restore = lambda value: value.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3)

        chunk_size = self.line_chunk_size if self.line_chunk_size > 0 else sequence.shape[0]
        outputs: list[torch.Tensor] = []
        for start in range(0, sequence.shape[0], chunk_size):
            end = min(start + chunk_size, sequence.shape[0])
            sequence_chunk = sequence[start:end]
            output_chunk = self._apply_mamba(forward_mamba, sequence_chunk)
            if self.bidirectional:
                if backward_mamba is None:
                    raise RuntimeError("bidirectional=True requires a backward Mamba module.")
                reversed_chunk = torch.flip(sequence_chunk, dims=[1])
                backward_output = self._apply_mamba(backward_mamba, reversed_chunk)
                output_chunk = output_chunk + torch.flip(backward_output, dims=[1])
            outputs.append(output_chunk)
        return restore(torch.cat(outputs, dim=0))

    def forward(
        self,
        x: torch.Tensor,
        base_condition: torch.Tensor,
        table_condition: torch.Tensor,
        table_reliability: torch.Tensor,
    ) -> torch.Tensor:
        gamma_base, beta_base, alpha_base = self.base_proj(base_condition).chunk(3, dim=1)
        gamma_table, beta_table, alpha_table = self.table_proj(table_condition).chunk(3, dim=1)
        reliability = table_reliability.to(dtype=x.dtype)
        gamma = gamma_base + reliability * gamma_table
        beta = beta_base + reliability * beta_table
        residual_gate = torch.sigmoid(alpha_base + reliability * alpha_table)

        h = x.permute(0, 2, 3, 4, 1)
        h = self.norm(h).permute(0, 4, 1, 2, 3)
        h = h * (1.0 + gamma[:, :, None, None, None].to(dtype=x.dtype))
        h = h + beta[:, :, None, None, None].to(dtype=x.dtype)

        modules = {
            "x": (0, self.x_forward, self.x_backward),
            "y": (1, self.y_forward, self.y_backward),
            "z": (2, self.z_forward, self.z_backward),
        }
        selected_indices = [self._AXIS_TO_INDEX[axis] for axis in self.active_axes]
        selected_weights = torch.softmax(self.axis_weight[selected_indices], dim=0)
        fused: Optional[torch.Tensor] = None
        for weight, axis_name in zip(selected_weights, self.active_axes):
            axis_index, forward_module, backward_module = modules[axis_name]
            axis_feature = self._run_axis(h, axis_index, forward_module, backward_module)
            weighted_feature = weight * axis_feature
            fused = weighted_feature if fused is None else fused + weighted_feature
            del axis_feature, weighted_feature
        if fused is None:
            raise RuntimeError("No active Mamba axis was configured.")
        fused = self.out_proj(fused)
        return (
            x
            + self.alpha.to(dtype=x.dtype)
            * residual_gate[:, :, None, None, None].to(dtype=x.dtype)
            * fused
        )


class FullResolutionGlobalTokenFusion3D(nn.Module):
    def __init__(
        self,
        local_channels: int,
        token_dim: int,
        attention_dim: int,
        num_heads: int,
        condition_dim: int,
        query_chunk: int = 2048,
    ) -> None:
        super().__init__()
        self.local_channels = int(local_channels)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        if self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads.")
        self.head_dim = self.attention_dim // self.num_heads
        self.query = nn.Conv3d(local_channels, self.attention_dim, kernel_size=1)
        self.key = nn.Linear(token_dim, self.attention_dim)
        self.value = nn.Linear(token_dim, self.attention_dim)
        self.out = nn.Conv3d(self.attention_dim, local_channels, kernel_size=1)
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(local_channels + self.attention_dim, local_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(local_channels, local_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.base_gate = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 1))
        self.table_gate = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 1))
        _zero_last_linear(self.table_gate)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.query_chunk = int(query_chunk)

    def forward(
        self,
        local_feature: torch.Tensor,
        global_tokens: torch.Tensor,
        base_condition: torch.Tensor,
        table_condition: torch.Tensor,
        table_reliability: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, depth, height, width = local_feature.shape
        query = self.query(local_feature).flatten(2).transpose(1, 2)
        key = self.key(global_tokens)
        value = self.value(global_tokens)
        query = query.view(batch, query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        if self.query_chunk <= 0:
            attention_output = _sdpa(query, key, value)
        else:
            chunks = []
            for start in range(0, query.shape[2], self.query_chunk):
                end = min(start + self.query_chunk, query.shape[2])
                chunks.append(_sdpa(query[:, :, start:end], key, value))
            attention_output = torch.cat(chunks, dim=2)

        attention_output = attention_output.transpose(1, 2).reshape(
            batch,
            depth * height * width,
            self.attention_dim,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch,
            self.attention_dim,
            depth,
            height,
            width,
        )
        projected = self.out(attention_output)
        spatial_gate = self.spatial_gate(torch.cat([local_feature, attention_output], dim=1))
        reliability_scalar = table_reliability.mean(dim=1, keepdim=True)
        fusion_strength = torch.sigmoid(
            self.base_gate(base_condition)
            + reliability_scalar * self.table_gate(table_condition)
        )
        return (
            local_feature
            + self.alpha.to(dtype=local_feature.dtype)
            * fusion_strength[:, :, None, None, None].to(dtype=local_feature.dtype)
            * spatial_gate
            * projected
        )


class GlobalCropFusion3D(nn.Module):
    def __init__(self, local_channels: int, global_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(global_channels, local_channels, kernel_size=1)
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(local_channels * 2, local_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(local_channels, local_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(local_channels * 2, max(local_channels // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(local_channels // 2, 16), local_channels),
            nn.Sigmoid(),
        )
        self.base_gate = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 1))
        self.table_gate = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 1))
        _zero_last_linear(self.table_gate)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        local_feature: torch.Tensor,
        global_crop: torch.Tensor,
        base_condition: torch.Tensor,
        table_condition: torch.Tensor,
        table_reliability: torch.Tensor,
    ) -> torch.Tensor:
        global_crop = self.proj(global_crop)
        spatial_gate = self.spatial_gate(torch.cat([local_feature, global_crop], dim=1))
        pooled = torch.cat(
            [local_feature.mean((2, 3, 4)), global_crop.mean((2, 3, 4))],
            dim=1,
        )
        channel_gate = self.channel_gate(pooled)[:, :, None, None, None]
        reliability_scalar = table_reliability.mean(dim=1, keepdim=True)
        fusion_strength = torch.sigmoid(
            self.base_gate(base_condition)
            + reliability_scalar * self.table_gate(table_condition)
        )
        return (
            local_feature
            + self.alpha.to(dtype=local_feature.dtype)
            * fusion_strength[:, :, None, None, None].to(dtype=local_feature.dtype)
            * channel_gate.to(dtype=local_feature.dtype)
            * spatial_gate
            * global_crop
        )


@dataclass
class GlobalContext:
    feature: torch.Tensor
    tokens: torch.Tensor
    time_embedding: torch.Tensor
    table_embedding: torch.Tensor
    table_condition: torch.Tensor
    table_mask: torch.Tensor
    case_reliability: torch.Tensor
    global_reliability: torch.Tensor


class GlobalLocalMambaPatchDenoiser3D(nn.Module):
    def __init__(
        self,
        table_num_features: int,
        image_size: tuple[int, int, int] = (160, 192, 160),
        patch_size: tuple[int, int, int] = (40, 48, 40),
        patch_grid: tuple[int, int, int] = (4, 4, 4),
        prediction_type: str = "residual",
        time_dim: int = 256,
        condition_dim: int = 512,
        table_dim: int = 256,
        position_dim: int = 128,
        global_xt_stem_channels: tuple[int, int, int] = (32, 64, 128),
        global_mri_stem_channels: tuple[int, int, int] = (32, 64, 128),
        global_output_channels: int = 128,
        global_mamba_depth: int = 6,
        global_mamba_d_state: int = 32,
        global_mamba_d_conv: int = 4,
        global_mamba_expand: int = 4,
        axial_bidirectional: bool = True,
        independent_axis_parameters: bool = True,
        independent_direction_parameters: bool = True,
        global_mamba_axes: tuple[str, ...] = ("x", "y", "z"),
        global_mamba_line_chunk_size: int = 0,
        global_mamba_gradient_checkpointing: bool = False,
        local_input_channels: int = 1,
        local_output_channels: int = 1,
        local_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        encoder_res_blocks: tuple[int, int, int] = (2, 2, 2),
        bottleneck_res_blocks: int = 4,
        decoder_res_blocks: tuple[int, int, int] = (2, 2, 2),
        norm_groups: int = 16,
        dropout: float = 0.1,
        token_grid: tuple[int, int, int] = (10, 12, 10),
        configured_num_tokens: Optional[int] = None,
        attention_dim: int = 128,
        token_heads: int = 8,
        full_resolution_query: bool = True,
        query_pooling: bool = False,
        query_chunk_size: int = 2048,
        table_encoder_type: str = "mixer",
        table_encoder_depth: int = 3,
        table_condition_dropout: float = 0.15,
        use_reliability_gate: bool = True,
        use_channelwise_reliability: bool = True,
        mri_structure_blocks_per_level: int = 2,
        mri_use_gradient: bool = True,
        mri_use_log: bool = True,
        zero_init_residual_scale: bool = True,
        separate_encoder_decoder_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(value) for value in image_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.prediction_type = str(prediction_type).lower()
        self.table_num_features = int(table_num_features)
        self.table_condition_dropout = float(table_condition_dropout)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.use_channelwise_reliability = bool(use_channelwise_reliability)
        self.mri_use_gradient = bool(mri_use_gradient)
        self.mri_use_log = bool(mri_use_log)
        self.global_output_channels = int(global_output_channels)

        if self.prediction_type != "residual":
            raise ValueError(
                "This model is paired with PatchResidualBridge and currently supports "
                "prediction_type='residual' only. Do not label x_t - PET as v-prediction."
            )
        if int(local_input_channels) != 1 or int(local_output_channels) != 1:
            raise ValueError("The Exp024 local denoiser requires one-channel Xt input and one-channel output.")
        expected_grid = tuple(
            image // patch for image, patch in zip(self.image_size, self.patch_size)
        )
        if expected_grid != self.patch_grid:
            raise ValueError(
                f"patch_grid={self.patch_grid} does not match image_size/patch_size={expected_grid}."
            )
        if not independent_axis_parameters:
            raise ValueError("This implementation currently requires independent axis parameters.")
        if axial_bidirectional and not independent_direction_parameters:
            raise ValueError("Bidirectional mode currently requires independent direction parameters.")
        if not full_resolution_query or query_pooling:
            raise ValueError(
                "The full Exp024 model requires full-resolution 40-scale queries and query_pooling=false."
            )
        if not separate_encoder_decoder_parameters:
            raise ValueError("Encoder and decoder fusion parameters must remain separate.")
        if not zero_init_residual_scale:
            raise ValueError("The formal Exp024 model requires zero-initialized residual fusion scales.")

        token_grid = tuple(int(value) for value in token_grid)
        actual_num_tokens = math.prod(token_grid)
        if configured_num_tokens is not None and int(configured_num_tokens) != actual_num_tokens:
            raise ValueError(
                f"num_tokens={configured_num_tokens} conflicts with token_grid={token_grid} "
                f"({actual_num_tokens} tokens)."
            )

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.table_encoder = TableEncoder(
            num_features=self.table_num_features,
            table_dim=table_dim,
            depth=table_encoder_depth,
            dropout=dropout,
            encoder_type=table_encoder_type,
        )
        self.table_to_condition = nn.Sequential(
            nn.Linear(table_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.position_encoder = PositionEncoder(position_dim, self.patch_grid)
        self.position_to_condition = nn.Sequential(
            nn.Linear(position_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        c40, c20, c10, c5 = [int(value) for value in local_channels]
        self.global_reliability = ReliabilityProjector(
            table_dim=table_dim,
            num_table_features=self.table_num_features,
            out_dims={"global": int(global_output_channels)},
        )
        self.local_reliability = ReliabilityProjector(
            table_dim=table_dim,
            num_table_features=self.table_num_features,
            out_dims={"40": c40, "20": c20, "10": c10, "5": c5},
            extra_dim=position_dim,
        )

        global_mri_input_channels = 1 + int(self.mri_use_gradient) + int(self.mri_use_log)
        self.global_xt_stem = GlobalStem(
            1,
            list(global_xt_stem_channels),
            norm_groups,
            dropout,
        )
        self.global_mri_stem = GlobalStem(
            global_mri_input_channels,
            list(global_mri_stem_channels),
            norm_groups,
            dropout,
        )
        self.global_fusion80 = DualGateFusion3D(
            global_xt_stem_channels[1],
            global_mri_stem_channels[1],
            norm_groups,
        )
        self.global_fusion40 = DualGateFusion3D(
            global_xt_stem_channels[2],
            global_mri_stem_channels[2],
            norm_groups,
        )
        self.global_down80_to40 = nn.Conv3d(
            global_xt_stem_channels[1],
            global_output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.global_direct40_proj = nn.Conv3d(
            global_xt_stem_channels[2],
            global_output_channels,
            kernel_size=1,
        )
        self.global_merge40 = nn.Sequential(
            nn.Conv3d(global_output_channels * 2, global_output_channels, kernel_size=1),
            ResidualConvBlock3D(global_output_channels, global_output_channels, norm_groups, dropout),
        )
        self.global_mamba_blocks = nn.ModuleList(
            [
                AxialBiMambaBlock(
                    dim=global_output_channels,
                    condition_dim=condition_dim,
                    d_state=global_mamba_d_state,
                    d_conv=global_mamba_d_conv,
                    expand=global_mamba_expand,
                    bidirectional=axial_bidirectional,
                    active_axes=tuple(global_mamba_axes),
                    line_chunk_size=global_mamba_line_chunk_size,
                    gradient_checkpointing=global_mamba_gradient_checkpointing,
                )
                for _ in range(int(global_mamba_depth))
            ]
        )

        self.token_grid = token_grid
        token_positions = self._build_token_positions(*self.token_grid)
        self.register_buffer("token_positions", token_positions, persistent=False)
        self.token_position_projection = nn.Linear(3, global_output_channels)

        self.local_mri_encoder = MRIStructureEncoder(
            channels=list(local_channels),
            norm_groups=norm_groups,
            dropout=dropout,
            blocks_per_level=int(mri_structure_blocks_per_level),
            use_gradient=self.mri_use_gradient,
            use_log=self.mri_use_log,
        )
        self.patch_stem = nn.Conv3d(1, c40, kernel_size=3, padding=1)
        self.encoder40 = nn.ModuleList(
            [
                ConditionalResBlock3D(c40, c40, condition_dim, condition_dim, norm_groups, dropout)
                for _ in range(int(encoder_res_blocks[0]))
            ]
        )
        self.down20 = Downsample3D(c40, c20)
        self.encoder20 = nn.ModuleList(
            [
                ConditionalResBlock3D(c20, c20, condition_dim, condition_dim, norm_groups, dropout)
                for _ in range(int(encoder_res_blocks[1]))
            ]
        )
        self.down10 = Downsample3D(c20, c10)
        self.encoder10 = nn.ModuleList(
            [
                ConditionalResBlock3D(c10, c10, condition_dim, condition_dim, norm_groups, dropout)
                for _ in range(int(encoder_res_blocks[2]))
            ]
        )
        self.down5 = Downsample3D(c10, c5)
        self.bottleneck = nn.ModuleList(
            [
                ConditionalResBlock3D(c5, c5, condition_dim, condition_dim, norm_groups, dropout)
                for _ in range(int(bottleneck_res_blocks))
            ]
        )
        self.up10 = Upsample3D(c5, c10)
        self.decoder10 = nn.ModuleList(
            [
                ConditionalResBlock3D(
                    c10 * 2 if index == 0 else c10,
                    c10,
                    condition_dim,
                    condition_dim,
                    norm_groups,
                    dropout,
                )
                for index in range(int(decoder_res_blocks[0]))
            ]
        )
        self.up20 = Upsample3D(c10, c20)
        self.decoder20 = nn.ModuleList(
            [
                ConditionalResBlock3D(
                    c20 * 2 if index == 0 else c20,
                    c20,
                    condition_dim,
                    condition_dim,
                    norm_groups,
                    dropout,
                )
                for index in range(int(decoder_res_blocks[1]))
            ]
        )
        self.up40 = Upsample3D(c20, c40)
        self.decoder40 = nn.ModuleList(
            [
                ConditionalResBlock3D(
                    c40 * 2 if index == 0 else c40,
                    c40,
                    condition_dim,
                    condition_dim,
                    norm_groups,
                    dropout,
                )
                for index in range(int(decoder_res_blocks[2]))
            ]
        )
        self.output_head = nn.Sequential(
            SafeGroupNorm(c40, norm_groups),
            nn.SiLU(),
            nn.Conv3d(c40, 1, kernel_size=3, padding=1),
        )

        self.encoder_mri40 = DualGateFusion3D(c40, c40, norm_groups)
        self.encoder_mri20 = DualGateFusion3D(c20, c20, norm_groups)
        self.encoder_mri10 = DualGateFusion3D(c10, c10, norm_groups)
        self.bottleneck_mri5 = DualGateFusion3D(c5, c5, norm_groups)
        self.decoder_mri10 = DualGateFusion3D(c10, c10, norm_groups)
        self.decoder_mri20 = DualGateFusion3D(c20, c20, norm_groups)
        self.decoder_mri40 = DualGateFusion3D(c40, c40, norm_groups)

        self.encoder_global_token40 = FullResolutionGlobalTokenFusion3D(
            c40,
            global_output_channels,
            attention_dim,
            token_heads,
            condition_dim,
            query_chunk=query_chunk_size,
        )
        self.decoder_global_token40 = FullResolutionGlobalTokenFusion3D(
            c40,
            global_output_channels,
            attention_dim,
            token_heads,
            condition_dim,
            query_chunk=query_chunk_size,
        )
        self.encoder_global_crop10 = GlobalCropFusion3D(c10, global_output_channels, condition_dim)
        self.decoder_global_crop10 = GlobalCropFusion3D(c10, global_output_channels, condition_dim)

    @staticmethod
    def _build_token_positions(depth: int, height: int, width: int) -> torch.Tensor:
        zz = torch.linspace(-1.0, 1.0, depth)
        yy = torch.linspace(-1.0, 1.0, height)
        xx = torch.linspace(-1.0, 1.0, width)
        mesh = torch.stack(torch.meshgrid(zz, yy, xx), dim=-1)
        return mesh.reshape(-1, 3)

    def _apply_table_condition_dropout(
        self,
        table: torch.Tensor,
        table_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.table_condition_dropout <= 0 or table.shape[1] == 0:
            return table, table_mask
        drop_case = (
            torch.rand(table.shape[0], 1, device=table.device)
            < self.table_condition_dropout
        )
        table = torch.where(drop_case, torch.zeros_like(table), table)
        table_mask = torch.where(drop_case, torch.ones_like(table_mask), table_mask)
        return table, table_mask

    def _prepare_table_inputs(
        self,
        reference: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = reference.shape[0]
        if table is None:
            table = reference.new_zeros((batch, self.table_num_features))
        if table_mask is None:
            table_mask = reference.new_ones((batch, self.table_num_features))
        if table.shape != table_mask.shape:
            raise ValueError(
                f"table and table_mask must have equal shape, got {table.shape} and {table_mask.shape}."
            )
        if table.shape[1] != self.table_num_features:
            raise ValueError(
                f"Expected {self.table_num_features} table features, got {table.shape[1]}."
            )
        return self._apply_table_condition_dropout(table, table_mask)

    def _global_tokens(self, global_feature: torch.Tensor) -> torch.Tensor:
        token_map = F.adaptive_avg_pool3d(global_feature, output_size=self.token_grid)
        tokens = token_map.flatten(2).transpose(1, 2)
        position = self.token_position_projection(
            self.token_positions.to(device=tokens.device, dtype=tokens.dtype)
        ).unsqueeze(0)
        return tokens + position

    def encode_global(
        self,
        x_t_full: torch.Tensor,
        mri_full: torch.Tensor,
        time: torch.Tensor,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
    ) -> GlobalContext:
        table, table_mask = self._prepare_table_inputs(x_t_full, table, table_mask)
        time_embedding = self.time_embedding(time)
        table_embedding, case_reliability = self.table_encoder(table, table_mask)
        table_condition = self.table_to_condition(table_embedding)
        if self.use_reliability_gate:
            global_reliability = self.global_reliability(
                table_embedding,
                case_reliability,
                table_mask,
            )["global"]
        else:
            global_reliability = torch.ones(
                x_t_full.shape[0],
                self.global_output_channels,
                device=x_t_full.device,
                dtype=x_t_full.dtype,
            )

        xt80, xt40 = self.global_xt_stem(x_t_full)
        mri_input = _build_mri_structure_input(
            mri_full,
            self.mri_use_gradient,
            self.mri_use_log,
        )
        mri80, mri40 = self.global_mri_stem(mri_input)
        fused80 = self.global_fusion80(xt80, mri80)
        fused40_direct = self.global_fusion40(xt40, mri40)
        fused40_from80 = self.global_down80_to40(fused80)
        fused40_direct = self.global_direct40_proj(fused40_direct)
        global_feature = self.global_merge40(
            torch.cat([fused40_from80, fused40_direct], dim=1)
        )
        for block in self.global_mamba_blocks:
            global_feature = block(
                global_feature,
                base_condition=time_embedding,
                table_condition=table_condition,
                table_reliability=global_reliability,
            )
        global_tokens = self._global_tokens(global_feature)
        return GlobalContext(
            feature=global_feature,
            tokens=global_tokens,
            time_embedding=time_embedding,
            table_embedding=table_embedding,
            table_condition=table_condition,
            table_mask=table_mask,
            case_reliability=case_reliability,
            global_reliability=global_reliability,
        )

    def _flatten_patches(
        self,
        x_t_patch: torch.Tensor,
        mri_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        global_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if x_t_patch.ndim == 6:
            batch, patches_per_case = x_t_patch.shape[:2]
            if batch != global_batch_size:
                raise ValueError(
                    f"Patch batch {batch} does not match global batch {global_batch_size}."
                )
            case_index = torch.arange(batch, device=x_t_patch.device).repeat_interleave(patches_per_case)
            return (
                x_t_patch.reshape(batch * patches_per_case, *x_t_patch.shape[2:]),
                mri_patch.reshape(batch * patches_per_case, *mri_patch.shape[2:]),
                patch_coords.reshape(batch * patches_per_case, 3),
                case_index,
                batch,
                patches_per_case,
            )
        if x_t_patch.ndim != 5:
            raise ValueError(
                f"x_t_patch must be [B,P,C,D,H,W] or [B,C,D,H,W], got {x_t_patch.shape}."
            )
        if x_t_patch.shape[0] != global_batch_size:
            raise ValueError(
                "A five-dimensional patch tensor is interpreted as one patch per global case. "
                "Use [B,P,C,D,H,W] for multiple patches per case."
            )
        batch = x_t_patch.shape[0]
        case_index = torch.arange(batch, device=x_t_patch.device)
        return x_t_patch, mri_patch, patch_coords.view(batch, 3), case_index, batch, 1

    def _extract_global_crop(
        self,
        global_feature: torch.Tensor,
        coords: torch.Tensor,
        case_index: torch.Tensor,
    ) -> torch.Tensor:
        """Extract fixed-grid global crops without Python loops or GPU .item() calls.

        This is mathematically equivalent to the original slicing implementation and
        does not change the network structure. It only reshapes the full global feature
        into a grid and gathers the crops requested by ``coords``.
        """
        batch, channels, global_depth, global_height, global_width = global_feature.shape
        grid_depth, grid_height, grid_width = self.patch_grid
        if (
            global_depth % grid_depth != 0
            or global_height % grid_height != 0
            or global_width % grid_width != 0
        ):
            raise ValueError("Global feature size is not divisible by patch_grid.")

        crop_depth = global_depth // grid_depth
        crop_height = global_height // grid_height
        crop_width = global_width // grid_width

        grid_feature = global_feature.reshape(
            batch,
            channels,
            grid_depth,
            crop_depth,
            grid_height,
            crop_height,
            grid_width,
            crop_width,
        )
        grid_feature = grid_feature.permute(
            0, 2, 4, 6, 1, 3, 5, 7
        ).contiguous().reshape(
            batch,
            grid_depth * grid_height * grid_width,
            channels,
            crop_depth,
            crop_height,
            crop_width,
        )

        coords = coords.long()
        flat_index = (
            coords[:, 0] * (grid_height * grid_width)
            + coords[:, 1] * grid_width
            + coords[:, 2]
        )
        return grid_feature[case_index.long(), flat_index]

    def denoise_patches(
        self,
        x_t_patch: torch.Tensor,
        mri_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        global_context: GlobalContext,
    ) -> torch.Tensor:
        (
            x_t_patch,
            mri_patch,
            patch_coords,
            case_index,
            batch_size,
            patches_per_case,
        ) = self._flatten_patches(
            x_t_patch,
            mri_patch,
            patch_coords,
            global_context.feature.shape[0],
        )
        position_embedding = self.position_encoder(patch_coords)
        base_condition = (
            global_context.time_embedding[case_index]
            + self.position_to_condition(position_embedding)
        )
        table_condition = global_context.table_condition[case_index]

        if self.use_reliability_gate:
            local_reliability = self.local_reliability(
                global_context.table_embedding[case_index],
                global_context.case_reliability[case_index],
                global_context.table_mask[case_index],
                extra=position_embedding,
            )
        else:
            local_reliability = {
                "40": torch.ones(x_t_patch.shape[0], self.patch_stem.out_channels, device=x_t_patch.device),
                "20": torch.ones(x_t_patch.shape[0], self.down20.conv.out_channels, device=x_t_patch.device),
                "10": torch.ones(x_t_patch.shape[0], self.down10.conv.out_channels, device=x_t_patch.device),
                "5": torch.ones(x_t_patch.shape[0], self.down5.conv.out_channels, device=x_t_patch.device),
            }
        if not self.use_channelwise_reliability:
            local_reliability = {
                key: value.mean(dim=1, keepdim=True).expand_as(value)
                for key, value in local_reliability.items()
            }

        mri_features = self.local_mri_encoder(mri_patch)
        global_tokens = global_context.tokens[case_index]
        global_crop = self._extract_global_crop(
            global_context.feature,
            patch_coords,
            case_index,
        )

        x = self.patch_stem(x_t_patch)
        for block in self.encoder40:
            x = block(x, base_condition, table_condition, local_reliability["40"])
        x = self.encoder_mri40(x, mri_features["40"])
        x = self.encoder_global_token40(
            x,
            global_tokens,
            base_condition,
            table_condition,
            local_reliability["40"],
        )
        skip40 = x

        x = self.down20(x)
        for block in self.encoder20:
            x = block(x, base_condition, table_condition, local_reliability["20"])
        x = self.encoder_mri20(x, mri_features["20"])
        skip20 = x

        x = self.down10(x)
        for block in self.encoder10:
            x = block(x, base_condition, table_condition, local_reliability["10"])
        x = self.encoder_mri10(x, mri_features["10"])
        x = self.encoder_global_crop10(
            x,
            global_crop,
            base_condition,
            table_condition,
            local_reliability["10"],
        )
        skip10 = x

        x = self.down5(x)
        middle_index = max(len(self.bottleneck) // 2 - 1, 0)
        for index, block in enumerate(self.bottleneck):
            x = block(x, base_condition, table_condition, local_reliability["5"])
            if index == middle_index:
                x = self.bottleneck_mri5(x, mri_features["5"])

        x = self.up10(x, skip10.shape[2:])
        x = torch.cat([x, skip10], dim=1)
        for block in self.decoder10:
            x = block(x, base_condition, table_condition, local_reliability["10"])
        x = self.decoder_mri10(x, mri_features["10"])
        x = self.decoder_global_crop10(
            x,
            global_crop,
            base_condition,
            table_condition,
            local_reliability["10"],
        )

        x = self.up20(x, skip20.shape[2:])
        x = torch.cat([x, skip20], dim=1)
        for block in self.decoder20:
            x = block(x, base_condition, table_condition, local_reliability["20"])
        x = self.decoder_mri20(x, mri_features["20"])

        x = self.up40(x, skip40.shape[2:])
        x = torch.cat([x, skip40], dim=1)
        for block in self.decoder40:
            x = block(x, base_condition, table_condition, local_reliability["40"])
        x = self.decoder_mri40(x, mri_features["40"])
        x = self.decoder_global_token40(
            x,
            global_tokens,
            base_condition,
            table_condition,
            local_reliability["40"],
        )
        output = self.output_head(x)
        if patches_per_case > 1:
            output = output.view(batch_size, patches_per_case, *output.shape[1:])
        return output

    def forward(
        self,
        x_t_full: torch.Tensor,
        mri_full: torch.Tensor,
        x_t_patch: torch.Tensor,
        mri_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        time: torch.Tensor,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        global_context = self.encode_global(
            x_t_full=x_t_full,
            mri_full=mri_full,
            time=time,
            table=table,
            table_mask=table_mask,
        )
        return self.denoise_patches(
            x_t_patch=x_t_patch,
            mri_patch=mri_patch,
            patch_coords=patch_coords,
            global_context=global_context,
        )
