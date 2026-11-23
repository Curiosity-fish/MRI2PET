"""Exp037B_v3_T6: T5 deep hierarchical BiCA + clinical semantic preservation.

T6 DOES NOT change the T5 MRI-Clinical interaction architecture.

The hierarchical pathway remains:
    T0
     |
Enc1 <-> T0 -> T1
     |
Enc2 <-> T1 -> T2
     |
Mid  <-> T2 -> T3
     |
Dec2 <-> T3 -> T4
     |
Dec1 <-> T4 -> T5
     |
Dec0 <-  T5

T6 adds only one training-oriented semantic branch:
    final refined clinical tokens T5
        -> lightweight token-wise ClinicalSemanticDecoder
        -> reconstructed normalized clinical values c_hat

This supports HCSPL (Hierarchical Clinical Semantic Preservation Loss):
the final image-aware clinical representation is allowed to change, but it
must still preserve enough clinical semantics to reconstruct the observed
patient-level Table variables.

The clinical decoder is part of the Table parameter group and is not required
for PET inference.

Pure-regression setting remains unchanged:
    MRI -> avg-pool/repeat endpoint -> 3D U-Net -> PET latent
        -> frozen PET AE decoder -> PET

No diffusion / Mamba / BSFE / Global-Local / ROI loss is used in T6.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.exp034a_basic_unet3d import TimeEmbedding, TimeResBlock3D


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
            raise ValueError(
                f"Expected table [B,{self.num_features}], got {tuple(values.shape)}"
            )

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
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by heads ({heads})"
            )

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
        tokens, key_padding_mask, observed, reliability = self.tokenizer(
            table, table_mask
        )
        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        tokens = self.token_norm(tokens)

        weights = observed.to(tokens.dtype).unsqueeze(-1)
        pooled = (
            (tokens * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )
        subject = self.subject_mlp(
            torch.cat([pooled, reliability.to(pooled.dtype)], dim=1)
        )

        return {
            "tokens": tokens,
            "key_padding_mask": key_padding_mask,
            "effective_observed": observed,
            "reliability": reliability,
            "subject_token": subject,
        }


class HierarchicalBiCAStage3D(nn.Module):
    """One bidirectional interaction stage in the hierarchy.

    At hierarchy level i:
        dP_i   = CA_I<-T(P_i, T_i)
        dT_i   = CA_T<-I(T_i, P_i)
        P_i'   = P_i + projected(dP_i)
        T_i+1  = T_i + bounded(dT_i)

    The UPDATED clinical tokens T_i+1 are explicitly passed to the next level.

    table_reads_image_pool:
        Optional spatial average-pooling before image tokens are used as K/V
        for Table<-Image attention. This reduces memory at Enc1/Dec1 without
        weakening Image<-Table conditioning at the full feature resolution.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int,
        heads: int = 4,
        image_residual_scale: float = 0.10,
        image_max_injection_ratio: float = 0.02,
        table_update_scale: float = 0.05,
        table_max_update_ratio: float = 0.10,
        out_init_std: float = 5e-3,
        table_reads_image_pool: int = 1,
    ) -> None:
        super().__init__()

        if token_dim % heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by heads ({heads})"
            )
        if image_max_injection_ratio <= 0:
            raise ValueError("image_max_injection_ratio must be > 0")
        if table_max_update_ratio <= 0:
            raise ValueError("table_max_update_ratio must be > 0")
        if int(table_reads_image_pool) < 1:
            raise ValueError("table_reads_image_pool must be >= 1")

        self.channels = int(channels)
        self.token_dim = int(token_dim)
        self.image_residual_scale = float(image_residual_scale)
        self.image_max_injection_ratio = float(image_max_injection_ratio)
        self.table_update_scale = float(table_update_scale)
        self.table_max_update_ratio = float(table_max_update_ratio)
        self.table_reads_image_pool = int(table_reads_image_pool)

        self.image_to_tokens = nn.Conv3d(channels, token_dim, kernel_size=1)

        self.image_query_norm = nn.LayerNorm(token_dim)
        self.table_kv_norm = nn.LayerNorm(token_dim)
        self.table_query_norm = nn.LayerNorm(token_dim)
        self.image_kv_norm = nn.LayerNorm(token_dim)

        self.image_reads_table = nn.MultiheadAttention(
            token_dim,
            heads,
            batch_first=True,
        )
        self.table_reads_image = nn.MultiheadAttention(
            token_dim,
            heads,
            batch_first=True,
        )

        self.image_delta_norm = nn.LayerNorm(token_dim)
        self.table_delta_norm = nn.LayerNorm(token_dim)

        self.tokens_to_feature = nn.Conv3d(token_dim, channels, kernel_size=1)
        nn.init.normal_(
            self.tokens_to_feature.weight,
            mean=0.0,
            std=float(out_init_std),
        )
        nn.init.zeros_(self.tokens_to_feature.bias)

    @staticmethod
    def _stable_smooth_scale(
        ratio: torch.Tensor,
        max_ratio: float,
    ) -> torch.Tensor:
        """Stable tanh(x)/x scaling with no hidden 0/0 branch."""
        x = ratio / float(max_ratio)
        eps = 1e-6
        safe_x = x.clamp_min(eps)
        candidate = torch.tanh(x) / safe_x
        return torch.where(
            x > eps,
            candidate,
            torch.ones_like(x),
        )

    def _cap_image_delta(
        self,
        feature: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_norm = feature.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        delta_norm = delta.float().flatten(1).norm(dim=1)
        raw_ratio = delta_norm / feature_norm

        cap_scale = self._stable_smooth_scale(
            raw_ratio,
            self.image_max_injection_ratio,
        )
        delta = delta * cap_scale[:, None, None, None, None].to(delta.dtype)

        final_ratio = (
            delta.float().flatten(1).norm(dim=1)
            / feature_norm
        )
        return delta, raw_ratio, final_ratio, cap_scale

    def _cap_table_delta(
        self,
        table_tokens: torch.Tensor,
        table_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table_norm = table_tokens.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        delta_norm = table_delta.float().flatten(1).norm(dim=1)
        raw_ratio = delta_norm / table_norm

        cap_scale = self._stable_smooth_scale(
            raw_ratio,
            self.table_max_update_ratio,
        )
        table_delta = table_delta * cap_scale[:, None, None].to(table_delta.dtype)

        final_ratio = (
            table_delta.float().flatten(1).norm(dim=1)
            / table_norm
        )
        return table_delta, raw_ratio, final_ratio

    def _image_tokens_for_table_kv(
        self,
        image_grid: torch.Tensor,
    ) -> torch.Tensor:
        if self.table_reads_image_pool > 1:
            k = self.table_reads_image_pool
            image_grid = F.avg_pool3d(
                image_grid,
                kernel_size=k,
                stride=k,
            )
        return image_grid.flatten(2).transpose(1, 2)

    def forward(
        self,
        feature: torch.Tensor,
        table_tokens: torch.Tensor,
        table_key_padding_mask: torch.Tensor,
        effective_observed: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, _, d, h, w = feature.shape

        image_grid = self.image_to_tokens(feature)
        image_tokens = image_grid.flatten(2).transpose(1, 2)

        # Full-resolution Image <- Table.
        t_kv = self.table_kv_norm(table_tokens)
        image_from_table = self.image_reads_table(
            self.image_query_norm(image_tokens),
            t_kv,
            t_kv,
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )[0]

        # Potentially pooled Image K/V for Table <- Image.
        image_kv = self._image_tokens_for_table_kv(image_grid)
        image_kv = self.image_kv_norm(image_kv)
        table_from_image = self.table_reads_image(
            self.table_query_norm(table_tokens),
            image_kv,
            image_kv,
            need_weights=False,
        )[0]

        # ------------------------------------------------------------
        # Image update: only cross-modal delta -> projection -> reliability
        # gate -> smooth norm cap.
        # ------------------------------------------------------------
        image_delta_tokens = self.image_delta_norm(image_from_table)
        delta_grid = image_delta_tokens.transpose(1, 2).reshape(
            b, self.token_dim, d, h, w
        )
        raw_image_delta = (
            self.image_residual_scale
            * self.tokens_to_feature(delta_grid)
        )
        raw_image_delta = raw_image_delta * (
            reliability.to(raw_image_delta.dtype)[:, :, None, None, None]
        )

        image_delta, image_raw_ratio, image_ratio, image_cap_scale = (
            self._cap_image_delta(feature, raw_image_delta)
        )
        feature_next = feature + image_delta

        # ------------------------------------------------------------
        # Table update: only truly observed clinical tokens are modified.
        # The update itself is also bounded because T6 repeats it five times.
        # ------------------------------------------------------------
        observed_gate = effective_observed.to(table_from_image.dtype).unsqueeze(-1)
        raw_table_delta = (
            self.table_update_scale
            * self.table_delta_norm(table_from_image)
            * observed_gate
        )
        table_delta, table_raw_ratio, table_ratio = self._cap_table_delta(
            table_tokens,
            raw_table_delta,
        )
        table_next = table_tokens + table_delta

        aux = {
            "injection_ratio": image_ratio.mean(),
            "raw_injection_ratio": image_raw_ratio.mean(),
            "image_cap_scale": image_cap_scale.mean(),
            "table_update_ratio": table_ratio.mean(),
            "table_raw_update_ratio": table_raw_ratio.mean(),
            "image_from_table_norm": (
                image_from_table.float().flatten(1).norm(dim=1).mean()
            ),
            "table_from_image_norm": (
                table_from_image.float().flatten(1).norm(dim=1).mean()
            ),
            "updated_table_token_norm": (
                table_next.float().flatten(1).norm(dim=1).mean()
            ),
            "all_finite": torch.stack(
                [
                    torch.isfinite(feature_next).all(),
                    torch.isfinite(table_next).all(),
                    torch.isfinite(image_ratio).all(),
                    torch.isfinite(table_ratio).all(),
                    torch.isfinite(image_from_table).all(),
                    torch.isfinite(table_from_image).all(),
                ]
            ).all(),
        }
        return feature_next, table_next, aux


class TerminalTableToImage3D(nn.Module):
    """Final high-resolution Table -> Image feedback at Dec0.

    This is analogous to using the final refined clinical state once more after
    the hierarchical bidirectional stack. No new Table state is produced here.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int,
        heads: int = 4,
        residual_scale: float = 0.10,
        max_injection_ratio: float = 0.005,
        out_init_std: float = 5e-3,
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.token_dim = int(token_dim)
        self.residual_scale = float(residual_scale)
        self.max_injection_ratio = float(max_injection_ratio)

        self.image_to_tokens = nn.Conv3d(channels, token_dim, 1)
        self.image_norm = nn.LayerNorm(token_dim)
        self.table_norm = nn.LayerNorm(token_dim)

        self.image_reads_table = nn.MultiheadAttention(
            token_dim,
            heads,
            batch_first=True,
        )

        self.delta_norm = nn.LayerNorm(token_dim)
        self.tokens_to_feature = nn.Conv3d(token_dim, channels, 1)
        nn.init.normal_(
            self.tokens_to_feature.weight,
            mean=0.0,
            std=float(out_init_std),
        )
        nn.init.zeros_(self.tokens_to_feature.bias)

    def forward(
        self,
        feature: torch.Tensor,
        table_tokens: torch.Tensor,
        table_key_padding_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, _, d, h, w = feature.shape

        p = self.image_to_tokens(feature).flatten(2).transpose(1, 2)
        t = self.table_norm(table_tokens)

        image_from_table = self.image_reads_table(
            self.image_norm(p),
            t,
            t,
            key_padding_mask=table_key_padding_mask,
            need_weights=False,
        )[0]

        delta_tokens = self.delta_norm(image_from_table)
        delta_grid = delta_tokens.transpose(1, 2).reshape(
            b, self.token_dim, d, h, w
        )
        raw_delta = self.residual_scale * self.tokens_to_feature(delta_grid)
        raw_delta = raw_delta * (
            reliability.to(raw_delta.dtype)[:, :, None, None, None]
        )

        feature_norm = feature.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        raw_delta_norm = raw_delta.float().flatten(1).norm(dim=1)
        raw_ratio = raw_delta_norm / feature_norm

        x = raw_ratio / self.max_injection_ratio
        eps = 1e-6
        safe_x = x.clamp_min(eps)
        candidate = torch.tanh(x) / safe_x
        cap_scale = torch.where(
            x > eps,
            candidate,
            torch.ones_like(x),
        )

        delta = raw_delta * cap_scale[:, None, None, None, None].to(raw_delta.dtype)
        final_ratio = (
            delta.float().flatten(1).norm(dim=1)
            / feature_norm
        )
        out = feature + delta

        aux = {
            "injection_ratio": final_ratio.mean(),
            "raw_injection_ratio": raw_ratio.mean(),
            "image_cap_scale": cap_scale.mean(),
            "image_from_table_norm": (
                image_from_table.float().flatten(1).norm(dim=1).mean()
            ),
            "all_finite": torch.stack(
                [
                    torch.isfinite(out).all(),
                    torch.isfinite(final_ratio).all(),
                    torch.isfinite(image_from_table).all(),
                ]
            ).all(),
        }
        return out, aux



class ClinicalSemanticDecoder(nn.Module):
    """Lightweight token-wise decoder used only to preserve clinical semantics.

    Input:
        refined_table_tokens: [B, F, D]

    Output:
        reconstructed normalized clinical scalars: [B, F]

    The same small MLP is shared across feature tokens. Feature identity is
    already encoded in each token by the tokenizer's feature embedding.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.decoder = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Keep the initial auxiliary head gentle.
        final = self.decoder[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward(self, table_tokens: torch.Tensor) -> torch.Tensor:
        if table_tokens.ndim != 3:
            raise ValueError(
                f"Expected clinical tokens [B,F,D], got {tuple(table_tokens.shape)}"
            )
        return self.decoder(table_tokens).squeeze(-1)


class Exp037Bv3T6UNet3D(nn.Module):
    """Exp034A-compatible U-Net + five H-BiCA levels + terminal feedback."""

    TABLE_PREFIXES = (
        "table_encoder.",
        "hica_enc1.",
        "hica_enc2.",
        "hica_mid.",
        "hica_dec2.",
        "hica_dec1.",
        "terminal_dec0.",
        "clinical_decoder.",
    )

    # Everything from Enc1 onward is considered semantic/high-level for
    # progressive unfreezing. Enc0/input/time_embedding remain shallow.
    SEMANTIC_PREFIXES = (
        "down1.",
        "enc1a.",
        "enc1b.",
        "down2.",
        "enc2a.",
        "enc2b.",
        "down3.",
        "mid1.",
        "mid2.",
        "up2.",
        "dec2a.",
        "dec2b.",
        "up1.",
        "dec1a.",
        "dec1b.",
        "up0.",
        "dec0a.",
        "dec0b.",
        "out.",
    )

    HIERARCHY_NAMES = ("enc1", "enc2", "mid", "dec2", "dec1")

    def __init__(
        self,
        table_num_features: int,
        feature_type_ids: Sequence[int],
        in_channels: int = 4,
        out_channels: int = 4,
        channels: Sequence[int] = (96, 192, 384, 768),
        time_dim: int = 384,
        norm_groups: int = 8,
        dropout: float = 0.0,
        table_token_dim: int = 128,
        table_depth: int = 2,
        table_heads: int = 4,
        table_ff_mult: int = 4,
        table_dropout: float = 0.0,
        table_num_feature_types: int = 5,
        bica_heads: int = 4,
        bica_out_init_std: float = 5e-3,
        enc1_max_injection_ratio: float = 0.01,
        enc1_table_update_scale: float = 0.05,
        enc2_max_injection_ratio: float = 0.02,
        enc2_table_update_scale: float = 0.075,
        mid_max_injection_ratio: float = 0.04,
        mid_table_update_scale: float = 0.10,
        dec2_max_injection_ratio: float = 0.02,
        dec2_table_update_scale: float = 0.075,
        dec1_max_injection_ratio: float = 0.01,
        dec1_table_update_scale: float = 0.05,
        terminal_dec0_max_injection_ratio: float = 0.005,
        table_max_update_ratio: float = 0.10,
        image_residual_scale: float = 0.10,
        clinical_decoder_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        c0, c1, c2, c3 = [int(v) for v in channels]

        # Exact Exp034A backbone names for checkpoint compatibility.
        self.time_embedding = TimeEmbedding(time_dim)
        self.input = nn.Conv3d(in_channels, c0, 3, padding=1)

        self.enc0a = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.enc0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)

        self.enc1a = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.enc1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)

        self.enc2a = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.enc2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)

        self.mid1 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)
        self.mid2 = TimeResBlock3D(c3, c3, time_dim, norm_groups, dropout)

        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.dec2a = TimeResBlock3D(c2 * 2, c2, time_dim, norm_groups, dropout)
        self.dec2b = TimeResBlock3D(c2, c2, time_dim, norm_groups, dropout)

        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)
        self.dec1a = TimeResBlock3D(c1 * 2, c1, time_dim, norm_groups, dropout)
        self.dec1b = TimeResBlock3D(c1, c1, time_dim, norm_groups, dropout)

        self.up0 = nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1)
        self.dec0a = TimeResBlock3D(c0 * 2, c0, time_dim, norm_groups, dropout)
        self.dec0b = TimeResBlock3D(c0, c0, time_dim, norm_groups, dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(min(norm_groups, c0), c0),
            nn.SiLU(),
            nn.Conv3d(c0, out_channels, 3, padding=1),
        )

        self.table_encoder = ClinicalTableEncoder(
            num_features=table_num_features,
            feature_type_ids=feature_type_ids,
            token_dim=table_token_dim,
            depth=table_depth,
            heads=table_heads,
            ff_mult=table_ff_mult,
            dropout=table_dropout,
            num_feature_types=table_num_feature_types,
        )

        common = dict(
            token_dim=table_token_dim,
            heads=bica_heads,
            image_residual_scale=image_residual_scale,
            table_max_update_ratio=table_max_update_ratio,
            out_init_std=bica_out_init_std,
        )

        # Enc1/Dec1 use pooled Image K/V for the Table<-Image direction.
        self.hica_enc1 = HierarchicalBiCAStage3D(
            channels=c1,
            image_max_injection_ratio=enc1_max_injection_ratio,
            table_update_scale=enc1_table_update_scale,
            table_reads_image_pool=2,
            **common,
        )
        self.hica_enc2 = HierarchicalBiCAStage3D(
            channels=c2,
            image_max_injection_ratio=enc2_max_injection_ratio,
            table_update_scale=enc2_table_update_scale,
            table_reads_image_pool=1,
            **common,
        )
        self.hica_mid = HierarchicalBiCAStage3D(
            channels=c3,
            image_max_injection_ratio=mid_max_injection_ratio,
            table_update_scale=mid_table_update_scale,
            table_reads_image_pool=1,
            **common,
        )
        self.hica_dec2 = HierarchicalBiCAStage3D(
            channels=c2,
            image_max_injection_ratio=dec2_max_injection_ratio,
            table_update_scale=dec2_table_update_scale,
            table_reads_image_pool=1,
            **common,
        )
        self.hica_dec1 = HierarchicalBiCAStage3D(
            channels=c1,
            image_max_injection_ratio=dec1_max_injection_ratio,
            table_update_scale=dec1_table_update_scale,
            table_reads_image_pool=2,
            **common,
        )

        self.terminal_dec0 = TerminalTableToImage3D(
            channels=c0,
            token_dim=table_token_dim,
            heads=bica_heads,
            residual_scale=image_residual_scale,
            max_injection_ratio=terminal_dec0_max_injection_ratio,
            out_init_std=bica_out_init_std,
        )

        # T6-only auxiliary semantic head. It reads the FINAL refined clinical
        # tokens after Enc1->Enc2->Mid->Dec2->Dec1 interactions.
        self.clinical_decoder = ClinicalSemanticDecoder(
            token_dim=table_token_dim,
            hidden_dim=clinical_decoder_hidden_dim,
        )

    def parameter_group_name(self, name: str) -> str:
        if name.startswith(self.TABLE_PREFIXES):
            return "table"
        if name.startswith(self.SEMANTIC_PREFIXES):
            return "semantic"
        return "shallow"

    def split_parameter_groups(self):
        groups = {"table": [], "semantic": [], "shallow": []}
        for name, param in self.named_parameters():
            groups[self.parameter_group_name(name)].append(param)
        return groups

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"table_warmup", "semantic_finetune", "full_finetune"}:
            raise ValueError(f"Unknown stage: {stage}")

        for name, param in self.named_parameters():
            group = self.parameter_group_name(name)

            if stage == "table_warmup":
                trainable = group == "table"
            elif stage == "semantic_finetune":
                trainable = group in {"table", "semantic"}
            else:
                trainable = True

            param.requires_grad_(trainable)

    @staticmethod
    def _disabled_hica_aux(feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "injection_ratio": feature.new_zeros(()),
            "raw_injection_ratio": feature.new_zeros(()),
            "image_cap_scale": feature.new_ones(()),
            "table_update_ratio": feature.new_zeros(()),
            "table_raw_update_ratio": feature.new_zeros(()),
            "image_from_table_norm": feature.new_zeros(()),
            "table_from_image_norm": feature.new_zeros(()),
            "updated_table_token_norm": feature.new_zeros(()),
            "all_finite": torch.tensor(True, device=feature.device),
        }

    @staticmethod
    def _disabled_terminal_aux(feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "injection_ratio": feature.new_zeros(()),
            "raw_injection_ratio": feature.new_zeros(()),
            "image_cap_scale": feature.new_ones(()),
            "image_from_table_norm": feature.new_zeros(()),
            "all_finite": torch.tensor(True, device=feature.device),
        }

    def _run_hica(
        self,
        module: HierarchicalBiCAStage3D,
        feature: torch.Tensor,
        table_tokens: torch.Tensor,
        table_state: dict[str, torch.Tensor],
    ):
        return module(
            feature,
            table_tokens,
            table_state["key_padding_mask"],
            table_state["effective_observed"],
            table_state["reliability"],
        )

    def forward(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        table: torch.Tensor,
        table_mask: torch.Tensor,
        *,
        return_aux: bool = False,
        disable_table: bool = False,
    ):
        te = self.time_embedding(t)

        # ------------------------------------------------------------
        # Enc0: deliberately no Table interaction.
        # ------------------------------------------------------------
        s0 = self.enc0b(self.enc0a(self.input(x1), te), te)

        # ------------------------------------------------------------
        # Enc1 <-> T0 -> T1
        # ------------------------------------------------------------
        s1 = self.enc1b(self.enc1a(self.down1(s0), te), te)

        if disable_table:
            table_state = None
            table_tokens = None
            enc1_aux = self._disabled_hica_aux(s1)
        else:
            table_state = self.table_encoder(table, table_mask)
            s1, table_tokens, enc1_aux = self._run_hica(
                self.hica_enc1,
                s1,
                table_state["tokens"],
                table_state,
            )

        # ------------------------------------------------------------
        # Enc2 <-> T1 -> T2
        # ------------------------------------------------------------
        s2 = self.enc2b(self.enc2a(self.down2(s1), te), te)

        if disable_table:
            enc2_aux = self._disabled_hica_aux(s2)
        else:
            s2, table_tokens, enc2_aux = self._run_hica(
                self.hica_enc2,
                s2,
                table_tokens,
                table_state,
            )

        # ------------------------------------------------------------
        # Bottleneck <-> T2 -> T3
        # ------------------------------------------------------------
        h = self.mid2(self.mid1(self.down3(s2), te), te)

        if disable_table:
            mid_aux = self._disabled_hica_aux(h)
        else:
            h, table_tokens, mid_aux = self._run_hica(
                self.hica_mid,
                h,
                table_tokens,
                table_state,
            )

        # ------------------------------------------------------------
        # Dec2 <-> T3 -> T4
        # ------------------------------------------------------------
        h = self.dec2b(
            self.dec2a(torch.cat([self.up2(h), s2], dim=1), te),
            te,
        )

        if disable_table:
            dec2_aux = self._disabled_hica_aux(h)
        else:
            h, table_tokens, dec2_aux = self._run_hica(
                self.hica_dec2,
                h,
                table_tokens,
                table_state,
            )

        # ------------------------------------------------------------
        # Dec1 <-> T4 -> T5
        # ------------------------------------------------------------
        h = self.dec1b(
            self.dec1a(torch.cat([self.up1(h), s1], dim=1), te),
            te,
        )

        if disable_table:
            dec1_aux = self._disabled_hica_aux(h)
        else:
            h, table_tokens, dec1_aux = self._run_hica(
                self.hica_dec1,
                h,
                table_tokens,
                table_state,
            )

        # ------------------------------------------------------------
        # Dec0 <- final refined T5.
        # ------------------------------------------------------------
        h = self.dec0b(
            self.dec0a(torch.cat([self.up0(h), s0], dim=1), te),
            te,
        )

        if disable_table:
            terminal_aux = self._disabled_terminal_aux(h)
        else:
            h, terminal_aux = self.terminal_dec0(
                h,
                table_tokens,
                table_state["key_padding_mask"],
                table_state["reliability"],
            )

        out = self.out(h)

        # T6 semantic prediction is only needed when diagnostics/losses are
        # requested. PET inference can call return_aux=False and incurs no
        # ClinicalSemanticDecoder cost.
        if return_aux:
            if table_tokens is not None:
                clinical_pred = self.clinical_decoder(table_tokens)
            else:
                clinical_pred = torch.zeros_like(table)

            stage_aux = {
                "enc1": enc1_aux,
                "enc2": enc2_aux,
                "mid": mid_aux,
                "dec2": dec2_aux,
                "dec1": dec1_aux,
            }

            stage_ratios = torch.stack(
                [stage_aux[name]["injection_ratio"] for name in self.HIERARCHY_NAMES]
            )
            terminal_ratio = terminal_aux["injection_ratio"]

            aux = {
                "table_injection_ratio": torch.cat(
                    [stage_ratios, terminal_ratio[None]]
                ).mean(),
                "hierarchical_injection_ratio_max": torch.cat(
                    [stage_ratios, terminal_ratio[None]]
                ).max(),
                "bica_all_finite": (
                    torch.stack(
                        [stage_aux[name]["all_finite"] for name in self.HIERARCHY_NAMES]
                        + [terminal_aux["all_finite"]]
                    ).all()
                ),
                "subject_token_norm": (
                    table_state["subject_token"].float().norm(dim=1).mean()
                    if table_state is not None
                    else h.new_zeros(())
                ),
                "final_table_token_norm": (
                    table_tokens.float().flatten(1).norm(dim=1).mean()
                    if table_tokens is not None
                    else h.new_zeros(())
                ),
                "table_reliability": (
                    table_state["reliability"].float().mean()
                    if table_state is not None
                    else h.new_zeros(())
                ),

                # T6-only semantic-preservation outputs.
                "clinical_pred": clinical_pred,
                "clinical_pred_abs_mean": clinical_pred.float().abs().mean(),

                "terminal_dec0_injection_ratio": terminal_aux["injection_ratio"],
                "terminal_dec0_raw_injection_ratio": terminal_aux[
                    "raw_injection_ratio"
                ],
            }

            for name in self.HIERARCHY_NAMES:
                a = stage_aux[name]
                aux[f"{name}_injection_ratio"] = a["injection_ratio"]
                aux[f"{name}_raw_injection_ratio"] = a["raw_injection_ratio"]
                aux[f"{name}_table_update_ratio"] = a["table_update_ratio"]
                aux[f"{name}_table_raw_update_ratio"] = a[
                    "table_raw_update_ratio"
                ]

            return out, aux

        return out
