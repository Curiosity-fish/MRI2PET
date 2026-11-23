from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.residual_bridge_resunet3d import (
    Downsample3D,
    ResidualTimeBlock3D,
    SinusoidalTimeEmbedding,
    Upsample3D,
)
from models.resunet3d import SafeGroupNorm, SelfAttention3D, Swish


def _safe_heads(embed_dim: int, num_heads: int) -> int:
    heads = max(1, min(int(num_heads), int(embed_dim)))
    while heads > 1 and int(embed_dim) % heads != 0:
        heads -= 1
    return heads


class FeatureTableTokenizer(nn.Module):
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
        if feature_type_ids is None:
            feature_type_ids = [0] * self.num_features
        if len(feature_type_ids) != self.num_features:
            raise ValueError(
                f"feature_type_ids length mismatch: expected {self.num_features}, got {len(feature_type_ids)}"
            )
        self.register_buffer(
            "feature_type_ids",
            torch.tensor([int(v) for v in feature_type_ids], dtype=torch.long),
            persistent=False,
        )
        self.feature_embed = nn.Embedding(self.num_features, self.token_dim)
        self.type_embed = nn.Embedding(max(int(num_feature_types), 1), self.token_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(1, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.missing_embed = nn.Embedding(2, self.token_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(self.token_dim * 2 + 1, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.global_norm = nn.LayerNorm(self.token_dim)

    def forward(
        self,
        values: Optional[torch.Tensor],
        missing_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if values is None:
            return None, None, None, None
        if values.ndim != 2:
            raise ValueError(f"Expected table values [B,F], got {tuple(values.shape)}")
        batch, features = values.shape
        if features != self.num_features:
            raise ValueError(
                f"Table feature mismatch: configured {self.num_features}, got {features}"
            )
        if missing_mask is None:
            missing_mask = torch.zeros_like(values)
        missing = (missing_mask > 0.5).long().to(device=values.device)
        values = values.float().to(device=missing.device)
        feat_ids = torch.arange(self.num_features, device=values.device, dtype=torch.long)
        feat_embed = self.feature_embed(feat_ids)[None, :, :].expand(batch, -1, -1)
        type_embed = self.type_embed(self.feature_type_ids)[None, :, :].expand(batch, -1, -1)
        value_embed = self.value_proj(values.unsqueeze(-1))
        miss_embed = self.missing_embed(missing)
        tokens = self.token_norm(feat_embed + type_embed + value_embed + miss_embed)
        key_padding_mask = missing.bool()
        valid = (~key_padding_mask).float()
        reliability = valid.mean(dim=1, keepdim=True)
        pooled_valid = (tokens * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_all = tokens.mean(dim=1)
        global_token = self.global_norm(
            self.global_mlp(torch.cat([pooled_valid, pooled_all, reliability], dim=1))
        )
        return tokens, key_padding_mask, reliability, global_token


class SpatialTokenEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        token_grid: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.token_grid = tuple(int(v) for v in token_grid)
        self.num_tokens = int(self.token_grid[0] * self.token_grid[1] * self.token_grid[2])
        self.proj = nn.Linear(int(in_channels), int(token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, int(token_dim)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(int(token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(x, self.token_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.norm(self.proj(tokens) + self.pos_embed[:, : tokens.shape[1]])


class TokenCrossBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int) -> None:
        super().__init__()
        heads = _safe_heads(token_dim, num_heads)
        self.table_norm1 = nn.LayerNorm(token_dim)
        self.mri_norm1 = nn.LayerNorm(token_dim)
        self.table_from_mri = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.mri_from_table = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.table_norm2 = nn.LayerNorm(token_dim)
        self.mri_norm2 = nn.LayerNorm(token_dim)
        self.table_ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )
        self.mri_ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )

    def forward(
        self,
        table_tokens: torch.Tensor,
        table_key_padding_mask: Optional[torch.Tensor],
        mri_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table_delta, _ = self.table_from_mri(
            query=self.table_norm1(table_tokens),
            key=self.mri_norm1(mri_tokens),
            value=self.mri_norm1(mri_tokens),
            need_weights=False,
        )
        table_tokens = table_tokens + table_delta
        table_tokens = table_tokens + self.table_ff(self.table_norm2(table_tokens))

        mri_delta, _ = self.mri_from_table(
            query=self.mri_norm1(mri_tokens),
            key=self.table_norm1(table_tokens),
            value=self.table_norm1(table_tokens),
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )
        mri_tokens = mri_tokens + mri_delta
        mri_tokens = mri_tokens + self.mri_ff(self.mri_norm2(mri_tokens))
        return table_tokens, mri_tokens


class JointSubjectEncoder(nn.Module):
    def __init__(self, token_dim: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TokenCrossBlock(int(token_dim), int(num_heads)) for _ in range(int(depth))]
        )
        self.subject_mlp = nn.Sequential(
            nn.Linear(int(token_dim) * 3 + 1, int(token_dim) * 2),
            Swish(),
            nn.Linear(int(token_dim) * 2, int(token_dim)),
        )
        self.subject_norm = nn.LayerNorm(int(token_dim))

    def forward(
        self,
        table_tokens: Optional[torch.Tensor],
        table_key_padding_mask: Optional[torch.Tensor],
        mri_tokens: torch.Tensor,
        reliability: Optional[torch.Tensor],
        table_global: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if table_tokens is None:
            batch = mri_tokens.shape[0]
            subject = self.subject_norm(
                self.subject_mlp(
                    torch.cat(
                        [
                            mri_tokens.mean(dim=1),
                            mri_tokens.mean(dim=1),
                            mri_tokens.mean(dim=1),
                            mri_tokens.new_ones((batch, 1)),
                        ],
                        dim=1,
                    )
                )
            )
            return None, mri_tokens, subject
        if reliability is None:
            reliability = table_tokens.new_ones((table_tokens.shape[0], 1))
        for block in self.blocks:
            table_tokens, mri_tokens = block(table_tokens, table_key_padding_mask, mri_tokens)
        valid = (~table_key_padding_mask).float() if table_key_padding_mask is not None else torch.ones_like(table_tokens[..., 0])
        table_pooled = (table_tokens * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mri_pooled = mri_tokens.mean(dim=1)
        if table_global is None:
            table_global = table_pooled
        subject = self.subject_norm(
            self.subject_mlp(
                torch.cat([table_pooled, table_global, mri_pooled, reliability], dim=1)
            )
        )
        return table_tokens, mri_tokens, subject


class DensePriorGenerator3D(nn.Module):
    def __init__(self, cond_channels: int, token_dim: int) -> None:
        super().__init__()
        self.subject_proj = nn.Linear(int(token_dim), int(cond_channels))
        self.net = nn.Sequential(
            nn.Conv3d(int(cond_channels) * 2, int(cond_channels), kernel_size=3, padding=1),
            SafeGroupNorm(int(cond_channels), num_groups=1),
            Swish(),
            nn.Conv3d(int(cond_channels), int(cond_channels), kernel_size=3, padding=1),
        )

    def forward(
        self,
        condition: torch.Tensor,
        subject_token: torch.Tensor,
    ) -> torch.Tensor:
        subject_map = self.subject_proj(subject_token)[:, :, None, None, None].expand_as(condition)
        return self.net(torch.cat([condition, subject_map], dim=1))


class PromptInjector3D(nn.Module):
    def __init__(self, in_channels: int, prompt_channels: int, token_dim: int, time_dim: int) -> None:
        super().__init__()
        self.prompt_proj = nn.Conv3d(int(prompt_channels), int(in_channels), kernel_size=1)
        self.gate = nn.Sequential(
            nn.Linear(int(token_dim) + int(time_dim) + 1, int(in_channels)),
            Swish(),
            nn.Linear(int(in_channels), int(in_channels)),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor,
        subject_token: torch.Tensor,
        time_emb: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if prompt.shape[-3:] != x.shape[-3:]:
            prompt = F.interpolate(prompt, size=x.shape[-3:], mode="trilinear", align_corners=False)
        gate = self.gate(torch.cat([subject_token, time_emb, reliability], dim=1))
        return x + gate[:, :, None, None, None].to(dtype=x.dtype) * self.prompt_proj(prompt)


class TimeAwareBiCrossFusion3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        time_dim: int,
        image_heads: int,
        token_heads: int,
        reverse_pool_grid: tuple[int, int, int] = (4, 4, 4),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.reverse_pool_grid = tuple(int(v) for v in reverse_pool_grid)

        self.image_norm = SafeGroupNorm(self.in_channels, num_groups=16)
        self.image_q = nn.Linear(self.in_channels, self.in_channels)
        self.token_to_image = nn.Linear(self.token_dim, self.in_channels)
        self.image_attn = nn.MultiheadAttention(
            embed_dim=self.in_channels,
            num_heads=_safe_heads(self.in_channels, image_heads),
            batch_first=True,
        )
        self.image_out = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1)
        self.read_gate = nn.Sequential(
            nn.Linear(self.token_dim + int(time_dim) + 1, self.in_channels),
            Swish(),
            nn.Linear(self.in_channels, self.in_channels),
            nn.Sigmoid(),
        )

        self.reverse_proj = nn.Linear(self.in_channels, self.token_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=_safe_heads(self.token_dim, token_heads),
            batch_first=True,
        )
        self.token_ff = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim * 4),
            nn.GELU(),
            nn.Linear(self.token_dim * 4, self.token_dim),
        )
        self.write_gate = nn.Sequential(
            nn.Linear(self.token_dim + int(time_dim) + 2, self.token_dim),
            Swish(),
            nn.Linear(self.token_dim, self.token_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        table_tokens: Optional[torch.Tensor],
        table_key_padding_mask: Optional[torch.Tensor],
        subject_token: torch.Tensor,
        time_emb: torch.Tensor,
        reliability: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if table_tokens is None:
            return x, None
        batch, channels, depth, height, width = x.shape
        x_tokens = self.image_norm(x).flatten(2).transpose(1, 2)
        q = self.image_q(x_tokens)
        kv = self.token_to_image(table_tokens)
        image_delta, _ = self.image_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )
        image_delta = image_delta.transpose(1, 2).contiguous().view(batch, channels, depth, height, width)
        read_gate = self.read_gate(torch.cat([subject_token, time_emb, reliability], dim=1))
        x = x + read_gate[:, :, None, None, None].to(dtype=x.dtype) * self.image_out(image_delta)

        pooled = F.adaptive_avg_pool3d(self.image_norm(x), self.reverse_pool_grid)
        pooled_tokens = self.reverse_proj(pooled.flatten(2).transpose(1, 2))
        token_delta, _ = self.token_attn(
            query=self.token_norm(table_tokens),
            key=pooled_tokens,
            value=pooled_tokens,
            need_weights=False,
        )
        write_gate = self.write_gate(
            torch.cat([subject_token, time_emb, reliability, 1.0 - time_fraction], dim=1)
        )
        table_tokens = table_tokens + write_gate.unsqueeze(1).to(dtype=table_tokens.dtype) * token_delta
        table_tokens = table_tokens + self.token_ff(self.token_norm(table_tokens))
        return x, table_tokens


class BSFETableJointPriorLatentBridgeDenoiser3D(nn.Module):
    """BSFE latent bridge with time-aware table fusion and dense spatial prior prompting."""

    def __init__(
        self,
        latent_channels: int = 1,
        cond_channels: int = 1,
        table_num_features: int = 0,
        base_channels: int = 32,
        channel_mults: Iterable[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 16,
        use_attention: bool = True,
        attention_level: int = 2,
        num_heads: int = 4,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 4,
        table_joint_depth: int = 2,
        mri_token_grid: tuple[int, int, int] = (4, 6, 4),
        reverse_pool_grid: tuple[int, int, int] = (4, 4, 4),
        num_feature_types: int = 5,
        feature_type_ids: Optional[Sequence[int]] = None,
        max_timesteps: int = 100,
    ) -> None:
        super().__init__()
        mults = [int(m) for m in channel_mults]
        channels = [int(base_channels) * m for m in mults]
        if len(channels) < 2:
            raise ValueError("channel_mults must have at least two stages.")

        self.latent_channels = int(latent_channels)
        self.cond_channels = int(cond_channels)
        self.table_num_features = int(table_num_features)
        self.table_token_dim = int(table_token_dim)
        self.max_timesteps = max(int(max_timesteps), 2)
        self.use_table = self.table_num_features > 0

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        if self.use_table:
            self.table_tokenizer = FeatureTableTokenizer(
                num_features=self.table_num_features,
                token_dim=self.table_token_dim,
                feature_type_ids=feature_type_ids,
                num_feature_types=int(num_feature_types),
            )
            self.spatial_tokenizer = SpatialTokenEncoder3D(
                in_channels=self.cond_channels,
                token_dim=self.table_token_dim,
                token_grid=mri_token_grid,
            )
            self.joint_encoder = JointSubjectEncoder(
                token_dim=self.table_token_dim,
                depth=int(table_joint_depth),
                num_heads=int(table_cross_attn_heads),
            )
        else:
            self.table_tokenizer = None
            self.spatial_tokenizer = None
            self.joint_encoder = None

        self.prior_generator = DensePriorGenerator3D(
            cond_channels=self.cond_channels,
            token_dim=self.table_token_dim if self.use_table else int(time_dim),
        )

        self.input_conv = nn.Conv3d(
            self.latent_channels + self.cond_channels,
            channels[0],
            kernel_size=3,
            padding=1,
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_ch = channels[0]
        for idx, ch in enumerate(channels):
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
            if idx < len(channels) - 1:
                self.downsamples.append(Downsample3D(current_ch, channels[idx + 1]))
                current_ch = channels[idx + 1]

        self.mid_blocks = nn.ModuleList(
            [
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
                SelfAttention3D(current_ch, num_heads=num_heads, num_groups=num_groups) if use_attention else nn.Identity(),
                ResidualTimeBlock3D(current_ch, current_ch, time_dim=time_dim, dropout=dropout, num_groups=num_groups),
            ]
        )

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        current_ch = rev_channels[0]
        decoder_channels: list[int] = []
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
            decoder_channels.append(skip_ch)

        token_dim = self.table_token_dim if self.use_table else int(time_dim)
        self.input_prompt = PromptInjector3D(channels[0], self.cond_channels, token_dim, time_dim)
        self.mid_prompt = PromptInjector3D(rev_channels[0], self.cond_channels, token_dim, time_dim)
        self.decoder_prompt = PromptInjector3D(decoder_channels[0], self.cond_channels, token_dim, time_dim)

        if self.use_table:
            self.input_fusion = TimeAwareBiCrossFusion3D(
                in_channels=channels[0],
                token_dim=self.table_token_dim,
                time_dim=time_dim,
                image_heads=int(num_heads),
                token_heads=int(table_cross_attn_heads),
                reverse_pool_grid=reverse_pool_grid,
            )
            self.mid_fusion = TimeAwareBiCrossFusion3D(
                in_channels=rev_channels[0],
                token_dim=self.table_token_dim,
                time_dim=time_dim,
                image_heads=int(num_heads),
                token_heads=int(table_cross_attn_heads),
                reverse_pool_grid=reverse_pool_grid,
            )
            self.decoder_fusion = TimeAwareBiCrossFusion3D(
                in_channels=decoder_channels[0],
                token_dim=self.table_token_dim,
                time_dim=time_dim,
                image_heads=int(num_heads),
                token_heads=int(table_cross_attn_heads),
                reverse_pool_grid=reverse_pool_grid,
            )
        else:
            self.input_fusion = None
            self.mid_fusion = None
            self.decoder_fusion = None

        self.out_norm = SafeGroupNorm(channels[0], num_groups=num_groups)
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[0], self.latent_channels, kernel_size=3, padding=1)

    def _build_joint_context(
        self,
        condition: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        batch = condition.shape[0]
        device = condition.device
        dtype = condition.dtype
        if not self.use_table or self.table_tokenizer is None or self.spatial_tokenizer is None or self.joint_encoder is None:
            reliability = condition.new_ones((batch, 1))
            subject = condition.mean(dim=(2, 3, 4))
            return None, None, subject, reliability
        miss = table_mask if table_mask is not None else torch.zeros_like(table)
        table_tokens, table_kpm, reliability, table_global = self.table_tokenizer(table.float(), miss.float())
        mri_tokens = self.spatial_tokenizer(condition)
        table_tokens, _, subject = self.joint_encoder(table_tokens, table_kpm, mri_tokens, reliability, table_global)
        return table_tokens.to(device=device, dtype=dtype), table_kpm, subject.to(device=device, dtype=dtype), reliability.to(device=device, dtype=dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        table: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat([x_t, condition], dim=1)
        t_emb = self.time_mlp(t)
        time_fraction = (t.float() / float(self.max_timesteps - 1)).clamp(0.0, 1.0)[:, None]
        table_tokens, table_kpm, subject_token, reliability = self._build_joint_context(condition, table, table_mask)
        prior_map = self.prior_generator(condition, subject_token)

        x = self.input_conv(x)
        x = self.input_prompt(x, prior_map, subject_token, t_emb, reliability)
        if self.input_fusion is not None:
            x, table_tokens = self.input_fusion(
                x,
                table_tokens,
                table_kpm,
                subject_token,
                t_emb,
                reliability,
                time_fraction,
            )

        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        for block in self.mid_blocks:
            if isinstance(block, ResidualTimeBlock3D):
                x = block(x, t_emb)
            else:
                x = block(x)
        x = self.mid_prompt(x, prior_map, subject_token, t_emb, reliability)
        if self.mid_fusion is not None:
            x, table_tokens = self.mid_fusion(
                x,
                table_tokens,
                table_kpm,
                subject_token,
                t_emb,
                reliability,
                time_fraction,
            )

        skips = list(reversed(skips[:-1]))
        for idx, (upsample, blocks, skip) in enumerate(zip(self.upsamples, self.up_blocks, skips)):
            x = upsample(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                if isinstance(block, ResidualTimeBlock3D):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            if idx == 0:
                x = self.decoder_prompt(x, prior_map, subject_token, t_emb, reliability)
                if self.decoder_fusion is not None:
                    x, table_tokens = self.decoder_fusion(
                        x,
                        table_tokens,
                        table_kpm,
                        subject_token,
                        t_emb,
                        reliability,
                        time_fraction,
                    )

        return self.out_conv(self.out_act(self.out_norm(x)))
