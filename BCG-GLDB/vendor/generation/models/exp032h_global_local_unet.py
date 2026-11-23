from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from models.exp032g_mri_control_unet import AdaptiveControl3D, MRIPatchControlEncoder3D, zero_module
from models.residual_bridge_resunet3d import Downsample3D, ResidualTimeBlock3D, SinusoidalTimeEmbedding, Upsample3D
from models.resunet3d import SafeGroupNorm, SelfAttention3D, Swish
from models.global_local_mamba_patch_denoiser import ResidualMamba


class PositionMLP3D(nn.Module):
    def __init__(self, dim: int, grid_size: tuple[int, int, int] = (4, 4, 4)) -> None:
        super().__init__()
        coords = []
        for z in range(grid_size[0]):
            for y in range(grid_size[1]):
                for x in range(grid_size[2]):
                    coords.append([
                        -1.0 + 2.0 * x / max(grid_size[2] - 1, 1),
                        -1.0 + 2.0 * y / max(grid_size[1] - 1, 1),
                        -1.0 + 2.0 * z / max(grid_size[0] - 1, 1),
                    ])
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32), persistent=False)
        self.net = nn.Sequential(nn.Linear(3, dim), Swish(), nn.Linear(dim, dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        pos = self.net(self.coords)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)


class BiMambaTokens(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 2, d_state: int = 4, d_conv: int = 3, expand: int = 1) -> None:
        super().__init__()
        self.fwd = nn.ModuleList([ResidualMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(depth)])
        self.bwd = nn.ModuleList([ResidualMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fwd, bwd in zip(self.fwd, self.bwd):
            y_f = fwd(x)
            y_b = torch.flip(bwd(torch.flip(x, dims=[1])), dims=[1])
            x = 0.5 * (y_f + y_b)
        return self.norm(x)


class ViTTokens(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 2, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GlobalFeedback(nn.Module):
    def __init__(self, feature_channels: int, global_dim: int) -> None:
        super().__init__()
        self.delta = zero_module(nn.Linear(global_dim, feature_channels))
        self.gate = nn.Sequential(nn.LayerNorm(global_dim), nn.Linear(global_dim, feature_channels), nn.Sigmoid())
        self.last_stats: dict[str, float] = {}

    def forward(
        self,
        feature: torch.Tensor,
        token: torch.Tensor,
        enabled: bool = True,
        shuffle: bool = False,
        stat_prefix: str = "global",
    ) -> torch.Tensor:
        if shuffle:
            token = token[:, torch.randperm(token.shape[1], device=token.device)]
        delta = self.delta(token)[:, :, :, None, None, None]
        gate = self.gate(token)[:, :, :, None, None, None]
        self.last_stats[f"{stat_prefix}_delta_norm"] = float(delta.detach().norm(dim=2).mean().item())
        self.last_stats[f"{stat_prefix}_gate_mean"] = float(gate.detach().mean().item())
        self.last_stats[f"{stat_prefix}_gate_std"] = float(gate.detach().std(unbiased=False).item())
        if not enabled:
            return feature
        b, n, c, d, h, w = feature.shape
        return feature + gate.to(feature.dtype) * delta.to(feature.dtype).expand(b, n, c, d, h, w)


class Exp032HGlobalLocalUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        base_channels: int = 16,
        channel_mults: Iterable[int] = (1, 2, 4),
        time_dim: int = 256,
        dropout: float = 0.0,
        num_groups: int = 8,
        use_attention: bool = False,
        attention_level: int = 1,
        num_heads: int = 4,
        mri_base_channels: int = 8,
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        strategy: str = "mambagl",
        global_dim: int = 64,
        global_depth: int = 2,
        vit_heads: int = 4,
        vit_mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.strategy = str(strategy).lower()
        self.global_dim = int(global_dim)
        mults = [int(m) for m in channel_mults]
        self.channels = [int(base_channels) * m for m in mults]
        if len(self.channels) != 3:
            raise ValueError("Exp032h requires high/mid/low channel_mults.")

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input_conv = nn.Conv3d(in_channels, self.channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_ch = self.channels[0]
        for idx, ch in enumerate(self.channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualTimeBlock3D(current_ch, ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        ResidualTimeBlock3D(ch, ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        SelfAttention3D(ch, num_heads=num_heads, num_groups=num_groups)
                        if use_attention and idx == attention_level
                        else nn.Identity(),
                    ]
                )
            )
            current_ch = ch
            if idx < len(self.channels) - 1:
                self.downsamples.append(Downsample3D(current_ch, self.channels[idx + 1]))
                current_ch = self.channels[idx + 1]
        self.mid_blocks = nn.ModuleList(
            [
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                SelfAttention3D(current_ch, num_heads=num_heads, num_groups=num_groups) if use_attention else nn.Identity(),
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
            ]
        )
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(self.channels))
        current_ch = rev_channels[0]
        for idx, skip_ch in enumerate(rev_channels[1:]):
            level = len(rev_channels) - idx - 2
            self.upsamples.append(Upsample3D(current_ch, skip_ch))
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualTimeBlock3D(skip_ch * 2, skip_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        ResidualTimeBlock3D(skip_ch, skip_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                        SelfAttention3D(skip_ch, num_heads=num_heads, num_groups=num_groups)
                        if use_attention and level == attention_level
                        else nn.Identity(),
                    ]
                )
            )
            current_ch = skip_ch
        self.out_norm = SafeGroupNorm(self.channels[0], num_groups=num_groups)
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(self.channels[0], out_channels, kernel_size=3, padding=1)
        self.mri_encoder = MRIPatchControlEncoder3D(
            base_channels=mri_base_channels,
            out_channels=self.channels,
            norm_groups=num_groups,
            dropout=dropout,
        )
        self.control = AdaptiveControl3D(self.channels, gamma_scale=gamma_scale, beta_scale=beta_scale)

        self.pos = PositionMLP3D(self.global_dim)
        if self.strategy == "vitgl":
            self.global_backbone = ViTTokens(self.global_dim, depth=global_depth, num_heads=vit_heads, mlp_ratio=vit_mlp_ratio)
        else:
            self.global_backbone = BiMambaTokens(self.global_dim, depth=global_depth)
        self.to_global = nn.ModuleDict({k: nn.Linear(c, self.global_dim) for k, c in zip(["h", "m", "l"], self.channels)})
        self.feedback = nn.ModuleDict({k: GlobalFeedback(c, self.global_dim) for k, c in zip(["h", "m", "l"], self.channels)})
        self.scale_emb = nn.ParameterDict({k: nn.Parameter(torch.zeros(1, 1, self.global_dim)) for k in ["h", "m", "l"]})
        self.h_to_m = nn.Linear(self.global_dim, self.global_dim)
        self.m_to_l = nn.Linear(self.global_dim, self.global_dim)
        self.last_global_stats: dict[str, float] = {}

    def load_adacontrol_state(self, state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(matched)
        self.load_state_dict(own, strict=True)

    def _tokens(self, feature: torch.Tensor, level: str, prev: torch.Tensor | None = None) -> torch.Tensor:
        pooled = feature.mean(dim=(-1, -2, -3))
        tok = self.to_global[level](pooled) + self.pos(feature.shape[0])
        tok = tok + self.scale_emb[level]
        if prev is not None:
            tok = tok + prev
        return tok

    def _run_global(self, feature: torch.Tensor, level: str, enabled: bool, shuffle: bool, prev: torch.Tensor | None = None):
        tok = self._tokens(feature, level, prev=prev)
        out = self.global_backbone(tok)
        self.last_global_stats[f"{level}_token_norm"] = float(out.detach().norm(dim=-1).mean().item())
        feat = self.feedback[level](feature, out, enabled=enabled, shuffle=shuffle, stat_prefix=level)
        self.last_global_stats.update(self.feedback[level].last_stats)
        return feat, out

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        mri_patch: torch.Tensor,
        control_enabled: bool = True,
        global_enabled: bool = True,
        global_shuffle: bool = False,
    ) -> torch.Tensor:
        bpatch = x_t.shape[0]
        if bpatch % 64 != 0:
            raise ValueError(f"Expected flattened 64-patch batch, got {bpatch}.")
        b = bpatch // 64
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)
        mri = self.mri_encoder(mri_patch)
        x = self.input_conv(x)
        skips = []
        global_prev = None
        levels = ["h", "m", "l"]
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                x = block(x, t_emb) if isinstance(block, ResidualTimeBlock3D) else block(x)
            x = self.control(x, mri[levels[idx]], levels[idx], enabled=control_enabled)
            x_group = x.view(b, 64, x.shape[1], *x.shape[-3:])
            do_level = self.strategy == "bifusion" or levels[idx] == "l"
            if do_level:
                x_group, g = self._run_global(x_group, levels[idx], enabled=global_enabled, shuffle=global_shuffle, prev=global_prev)
                if levels[idx] == "h":
                    global_prev = self.h_to_m(g)
                elif levels[idx] == "m":
                    global_prev = self.m_to_l(g)
                x = x_group.reshape(bpatch, x_group.shape[2], *x_group.shape[-3:])
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        for block in self.mid_blocks:
            x = block(x, t_emb) if isinstance(block, ResidualTimeBlock3D) else block(x)
        skips = list(reversed(skips[:-1]))
        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, skips):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, t_emb) if isinstance(block, ResidualTimeBlock3D) else block(x)
        return self.out_conv(self.out_act(self.out_norm(x)))
