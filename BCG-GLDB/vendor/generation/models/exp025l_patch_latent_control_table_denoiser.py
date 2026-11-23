from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.bsfe_table_jointprior_latent_bridge_denoiser import (
    DensePriorGenerator3D,
    FeatureTableTokenizer,
    JointSubjectEncoder,
    PromptInjector3D,
    SpatialTokenEncoder3D,
    TimeAwareBiCrossFusion3D,
)
from models.exp025k_patch_latent_control_denoiser import (
    Exp025KPatchLatentControlDenoiser3D,
)


class Exp025LPatchLatentControlTableDenoiser3D(Exp025KPatchLatentControlDenoiser3D):
    def __init__(
        self,
        full_latent_size: tuple[int, int, int],
        patch_latent_size: tuple[int, int, int],
        latent_channels: int = 4,
        cond_channels: int = 4,
        mri_encoder_channels: tuple[int, int, int] = (8, 16, 32),
        global_channels: tuple[int, int] = (32, 64),
        local_channels: tuple[int, int] = (32, 64),
        time_dim: int = 256,
        pos_dim: int = 256,
        norm_groups: int = 8,
        dropout: float = 0.0,
        table_num_features: int = 0,
        table_token_dim: int = 128,
        table_cross_attn_heads: int = 4,
        table_joint_depth: int = 2,
        num_feature_types: int = 5,
        feature_type_ids: Optional[Sequence[int]] = None,
        reverse_pool_grid: tuple[int, int, int] = (4, 4, 4),
        max_timesteps: int = 100,
    ) -> None:
        super().__init__(
            full_latent_size=full_latent_size,
            patch_latent_size=patch_latent_size,
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            mri_encoder_channels=mri_encoder_channels,
            global_channels=global_channels,
            local_channels=local_channels,
            time_dim=time_dim,
            pos_dim=pos_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        self.table_num_features = int(table_num_features)
        self.use_table = self.table_num_features > 0
        self.table_token_dim = int(table_token_dim)
        self.max_timesteps = max(int(max_timesteps), 2)
        c10, c5 = [int(v) for v in local_channels]

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
                token_grid=(4, 6, 4),
            )
            self.joint_encoder = JointSubjectEncoder(
                token_dim=self.table_token_dim,
                depth=int(table_joint_depth),
                num_heads=int(table_cross_attn_heads),
            )
            self.prior_generator = DensePriorGenerator3D(
                cond_channels=self.cond_channels,
                token_dim=self.table_token_dim,
            )
            self.prompt10e = PromptInjector3D(c10, self.cond_channels, self.table_token_dim, time_dim)
            self.prompt5 = PromptInjector3D(c5, self.cond_channels, self.table_token_dim, time_dim)
            self.prompt10d = PromptInjector3D(c10, self.cond_channels, self.table_token_dim, time_dim)
            self.fusion10e = TimeAwareBiCrossFusion3D(
                in_channels=c10,
                token_dim=self.table_token_dim,
                time_dim=time_dim,
                image_heads=4,
                token_heads=int(table_cross_attn_heads),
                reverse_pool_grid=reverse_pool_grid,
            )
            self.fusion5 = TimeAwareBiCrossFusion3D(
                in_channels=c5,
                token_dim=self.table_token_dim,
                time_dim=time_dim,
                image_heads=4,
                token_heads=int(table_cross_attn_heads),
                reverse_pool_grid=reverse_pool_grid,
            )
            self.fusion10d = TimeAwareBiCrossFusion3D(
                in_channels=c10,
                token_dim=self.table_token_dim,
                time_dim=time_dim,
                image_heads=4,
                token_heads=int(table_cross_attn_heads),
                reverse_pool_grid=reverse_pool_grid,
            )
        else:
            self.table_tokenizer = None
            self.spatial_tokenizer = None
            self.joint_encoder = None
            self.prior_generator = None
            self.prompt10e = None
            self.prompt5 = None
            self.prompt10d = None
            self.fusion10e = None
            self.fusion5 = None
            self.fusion10d = None

    def build_table_context(
        self,
        cond_full: torch.Tensor,
        table: Optional[torch.Tensor],
        table_mask: Optional[torch.Tensor],
    ) -> Optional[dict[str, torch.Tensor]]:
        if not self.use_table or table is None or self.table_tokenizer is None:
            return None
        if table_mask is None:
            table_mask = torch.zeros_like(table)
        table_tokens, table_kpm, reliability, table_global = self.table_tokenizer(table.float(), table_mask.float())
        mri_tokens = self.spatial_tokenizer(cond_full)
        table_tokens, _, subject = self.joint_encoder(table_tokens, table_kpm, mri_tokens, reliability, table_global)
        prior_full = self.prior_generator(cond_full, subject)
        return {
            "table_tokens": table_tokens.to(device=cond_full.device, dtype=cond_full.dtype),
            "table_kpm": table_kpm,
            "subject": subject.to(device=cond_full.device, dtype=cond_full.dtype),
            "reliability": reliability.to(device=cond_full.device, dtype=cond_full.dtype),
            "prior_full": prior_full.to(device=cond_full.device, dtype=cond_full.dtype),
        }

    @staticmethod
    def _expand_rows(x: Optional[torch.Tensor], rows: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.shape[0] == rows:
            return x
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch 1 or {rows}, got {x.shape[0]}")
        return x.expand(rows, *x.shape[1:])

    def denoise_patches(
        self,
        x_t_patch: torch.Tensor,
        cond_patch: torch.Tensor,
        patch_coords: torch.Tensor,
        time: torch.Tensor,
        global_context: dict[str, torch.Tensor],
        table_context: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        cond_vec = self._global_condition(time, global_context["pooled"], patch_coords)
        local_control = self.local_control(cond_patch)
        global10 = self._crop_feature(global_context["40"], patch_coords)
        global5 = self._crop_feature(global_context["20"], patch_coords)

        prior_patch = None
        table_tokens = None
        table_kpm = None
        subject = None
        reliability = None
        time_fraction = None
        if table_context is not None:
            prior_patch = self._crop_feature(table_context["prior_full"], patch_coords)
            table_tokens = self._expand_rows(table_context["table_tokens"], x_t_patch.shape[0])
            table_kpm = self._expand_rows(table_context["table_kpm"], x_t_patch.shape[0])
            subject = self._expand_rows(table_context["subject"], x_t_patch.shape[0])
            reliability = self._expand_rows(table_context["reliability"], x_t_patch.shape[0])
            time_fraction = (time.float() / float(self.max_timesteps - 1)).clamp(0.0, 1.0)[:, None]
            time_fraction = self._expand_rows(time_fraction, x_t_patch.shape[0])

        x = self.local_stem(x_t_patch)
        x = self.enc10a(x, cond_vec)
        x = self.enc10b(x, cond_vec)
        if prior_patch is not None and self.prompt10e is not None:
            x = self.prompt10e(x, prior_patch, subject, cond_vec, reliability)
        if table_tokens is not None and self.fusion10e is not None:
            x, table_tokens = self.fusion10e(x, table_tokens, table_kpm, subject, cond_vec, reliability, time_fraction)
        x = self._inject(x, local_control["10"], self.gate10e, self.inject10e, cond_vec)
        x = self.fuse_global10e(x, global10)
        skip10 = x

        x = self.down5(x)
        x = self.mid5a(x, cond_vec)
        if prior_patch is not None and self.prompt5 is not None:
            x = self.prompt5(x, prior_patch, subject, cond_vec, reliability)
        if table_tokens is not None and self.fusion5 is not None:
            x, table_tokens = self.fusion5(x, table_tokens, table_kpm, subject, cond_vec, reliability, time_fraction)
        x = self._inject(x, local_control["5"], self.gate5, self.inject5, cond_vec)
        x = self.fuse_global5(x, global5)
        x = self.mid5b(x, cond_vec)

        x = self.up10(x, skip10.shape[-3:])
        x = self.dec10a(torch.cat([x, skip10], dim=1), cond_vec)
        x = self.dec10b(x, cond_vec)
        if prior_patch is not None and self.prompt10d is not None:
            x = self.prompt10d(x, prior_patch, subject, cond_vec, reliability)
        if table_tokens is not None and self.fusion10d is not None:
            x, _ = self.fusion10d(x, table_tokens, table_kpm, subject, cond_vec, reliability, time_fraction)
        x = self._inject(x, local_control["10"], self.gate10d, self.inject10d, cond_vec)
        x = self.fuse_global10d(x, global10)
        return self.out(x)
