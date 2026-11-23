"""Chapter 3: shared Patch-AE and the seven-stage BCG-GLDB residual network.

This is a new training graph. Old checkpoint tensors must not be silently loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


from mri2pet.models.clinical_attention import ClinicalTableEncoder
from mri2pet.models.unet_blocks import TimeEmbedding, TimeResBlock3D
from mri2pet.models.blocks import ResidualBlock3D
from mri2pet.models.global_context import GlobalViT3D
from mri2pet.models.autoencoder import AE
from mri2pet.data.patches import build_patch_grid, extract_patches, stitch_patches


class SharedPatchAE(nn.Module):
    """One encoder/decoder is shared by 64 non-overlapping PET patches."""

    def __init__(
        self,
        image_shape=(160, 192, 160),
        patch_shape=(40, 48, 40),
        latent_channels=4,
        base_channels=16,
        chunk_size=16,
    ):
        super().__init__()
        self.image_shape = tuple(map(int, image_shape))
        self.patch_shape = tuple(map(int, patch_shape))
        self.latent_channels = int(latent_channels)
        self.chunk_size = int(chunk_size)
        if any(a % b for a, b in zip(self.image_shape, self.patch_shape)):
            raise ValueError("The image must be divisible into equal non-overlapping patches")
        if any(b % 4 for b in self.patch_shape):
            raise ValueError("Patch dimensions must be divisible by the AE factor of four")
        self.patch_coords = build_patch_grid(self.image_shape, self.patch_shape)
        if len(self.patch_coords) != 64:
            raise ValueError(f"The thesis requires 64 patches, found {len(self.patch_coords)}")
        self.latent_shape = tuple(s // 4 for s in self.image_shape)
        self.latent_patch_shape = tuple(s // 4 for s in self.patch_shape)
        self.latent_coords = build_patch_grid(self.latent_shape, self.latent_patch_shape)
        self.net = AE(
            image_channels=1,
            latent_channels=self.latent_channels,
            base_channels=base_channels,
            norm_groups=8,
            dropout=0.0,
        )

    def encode_volume(self, pet: torch.Tensor) -> torch.Tensor:
        if tuple(pet.shape[1:]) != (1, *self.image_shape):
            raise ValueError(f"Expected PET [B,1,{self.image_shape}], got {tuple(pet.shape)}")
        patches = extract_patches(pet, self.patch_coords, self.patch_shape)
        chunks = []
        for start in range(0, 64, self.chunk_size):
            part = patches[:, start : start + self.chunk_size]
            latent = self.net.encode(part.reshape(-1, 1, *self.patch_shape))
            chunks.append(
                latent.reshape(
                    pet.shape[0], part.shape[1], self.latent_channels, *self.latent_patch_shape
                )
            )
        return stitch_patches(torch.cat(chunks, dim=1), self.latent_coords, self.latent_shape)

    def decode_volume(self, latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape[1:]) != (self.latent_channels, *self.latent_shape):
            raise ValueError(f"Unexpected PET latent shape: {tuple(latent.shape)}")
        patches = extract_patches(latent, self.latent_coords, self.latent_patch_shape)
        chunks = []
        for start in range(0, 64, self.chunk_size):
            part = patches[:, start : start + self.chunk_size]
            decoded = self.net.decoder(
                part.reshape(-1, self.latent_channels, *self.latent_patch_shape)
            )
            chunks.append(decoded.reshape(latent.shape[0], part.shape[1], 1, *self.patch_shape))
        return stitch_patches(torch.cat(chunks, dim=1), self.patch_coords, self.image_shape)

    def forward(self, pet: torch.Tensor) -> torch.Tensor:
        return self.decode_volume(self.encode_volume(pet))

    def freeze_for_bridge(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class MRIResidualPyramid(nn.Module):
    """Four-stage MRI encoder after the specified fourfold average pooling."""

    def __init__(self, channels=(48, 96, 192, 384), groups=8):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = nn.Conv3d(1, c0, 3, padding=1)
        self.levels = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock3D(c0, c0, groups, 0.0), ResidualBlock3D(c0, c0, groups, 0.0)
                ),
                nn.Sequential(
                    nn.Conv3d(c0, c1, 3, stride=2, padding=1), ResidualBlock3D(c1, c1, groups, 0.0)
                ),
                nn.Sequential(
                    nn.Conv3d(c1, c2, 3, stride=2, padding=1), ResidualBlock3D(c2, c2, groups, 0.0)
                ),
                nn.Sequential(
                    nn.Conv3d(c2, c3, 3, stride=2, padding=1), ResidualBlock3D(c3, c3, groups, 0.0)
                ),
            ]
        )

    def forward(self, mri: torch.Tensor) -> list[torch.Tensor]:
        feature = self.stem(F.avg_pool3d(mri, kernel_size=4, stride=4))
        outputs = []
        for level in self.levels:
            feature = level(feature)
            outputs.append(feature)
        return outputs


class TimeGatedBiCA(nn.Module):
    """Image→table→image interaction at one of the seven U-Net locations."""

    def __init__(self, channels: int, token_dim=128, time_dim=384, heads=4):
        super().__init__()
        self.token_dim = token_dim
        self.image_projection = nn.Conv3d(channels, token_dim, 1)
        self.to_feature = nn.Conv3d(token_dim, channels, 1)
        self.time_to_table = nn.Linear(time_dim, token_dim)
        self.time_to_image = nn.Linear(time_dim, channels)
        self.query_norm = nn.LayerNorm(token_dim)
        self.table_norm = nn.LayerNorm(token_dim)
        self.delta_norm = nn.LayerNorm(token_dim)
        self.image_reads_table = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.table_reads_image = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.gate = nn.Conv3d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, feature, table_tokens, key_padding_mask, observed, reliability, time_embedding
    ):
        batch, _, depth, height, width = feature.shape
        grid = self.image_projection(feature)
        image_tokens = grid.flatten(2).transpose(1, 2)
        conditioned_table = table_tokens + self.time_to_table(time_embedding)[:, None, :]
        table_kv = self.table_norm(conditioned_table)
        first = self.image_reads_table(
            self.query_norm(image_tokens),
            table_kv,
            table_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        temporary_image = image_tokens + first
        image_kv = self.query_norm(temporary_image)
        table_delta = self.table_reads_image(
            self.table_norm(conditioned_table), image_kv, image_kv, need_weights=False
        )[0]
        updated_table = table_tokens + table_delta * observed[:, :, None].to(table_delta.dtype)
        final_table_kv = self.table_norm(
            updated_table + self.time_to_table(time_embedding)[:, None, :]
        )
        second = self.image_reads_table(
            self.query_norm(temporary_image),
            final_table_kv,
            final_table_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        delta = (
            self.delta_norm(first + second)
            .transpose(1, 2)
            .reshape(batch, self.token_dim, depth, height, width)
        )
        delta = self.to_feature(delta)
        gate_input = feature + self.time_to_image(time_embedding)[:, :, None, None, None]
        update = self.residual_scale * torch.sigmoid(self.gate(gate_input)) * delta
        update = update * reliability[:, :, None, None, None].to(update.dtype)
        return feature + update, updated_table


class GlobalContextPyramid(nn.Module):
    """Global branch reads the current bridge state, then returns four scales."""

    def __init__(
        self,
        in_channels=4,
        channels=(96, 192, 384, 768),
        token_dim=256,
        depth=2,
        heads=4,
        bottleneck_shape=(5, 6, 5),
    ):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.down = nn.ModuleList(
            [nn.Conv3d(a, b, 3, stride=2, padding=1) for a, b in ((c0, c1), (c1, c2), (c2, c3))]
        )
        self.transformer = GlobalViT3D(
            c3, token_dim=token_dim, depth=depth, heads=heads, bottleneck_shape=bottleneck_shape
        )
        self.up = nn.ModuleList([nn.Conv3d(c3, c2, 1), nn.Conv3d(c2, c1, 1), nn.Conv3d(c1, c0, 1)])

    def forward(self, bridge_state):
        h = self.stem(bridge_state)
        shapes = [h.shape[-3:]]
        for layer in self.down:
            h = layer(h)
            shapes.append(h.shape[-3:])
        h = self.transformer(h)
        outputs = [None, None, None, h]
        for index, projection in zip((2, 1, 0), self.up):
            h = projection(
                F.interpolate(h, size=shapes[index], mode="trilinear", align_corners=False)
            )
            outputs[index] = h
        return outputs


class GlobalResidualGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv3d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv3d(channels * 2, channels, 1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, local, global_feature):
        if local.shape[-3:] != global_feature.shape[-3:]:
            global_feature = F.interpolate(
                global_feature, size=local.shape[-3:], mode="trilinear", align_corners=False
            )
        projected = self.projection(global_feature)
        return local + self.scale * self.gate(torch.cat([local, projected], dim=1)) * projected


@dataclass(frozen=True)
class GenerationShape:
    image: tuple[int, int, int] = (160, 192, 160)
    latent: tuple[int, int, int] = (40, 48, 40)


class PaperBCGGLDB(nn.Module):
    """Residual predictor for z_k - z_P; exactly seven clinical interactions."""

    STAGES = ("enc0", "enc1", "enc2", "mid", "dec2", "dec1", "dec0")
    MRI_LEVELS = (0, 1, 2, 3, 2, 1, 0)

    def __init__(
        self,
        table_feature_types: list[int],
        channels=(96, 192, 384, 768),
        mri_channels=(48, 96, 192, 384),
        time_dim=384,
        token_dim=128,
        global_token_dim=256,
        global_depth=2,
        global_heads=4,
        bottleneck_shape=(5, 6, 5),
        use_mri_structure=True,
        use_clinical=True,
        use_global=True,
    ):
        super().__init__()
        self.use_mri_structure = bool(use_mri_structure)
        self.use_clinical = bool(use_clinical)
        self.use_global = bool(use_global)
        c0, c1, c2, c3 = channels
        self.time = TimeEmbedding(time_dim)
        self.mri_encoder = MRIResidualPyramid(mri_channels)
        self.table_encoder = ClinicalTableEncoder(
            len(table_feature_types),
            table_feature_types,
            token_dim=token_dim,
            depth=2,
            heads=4,
            ff_mult=4,
            dropout=0.0,
        )
        self.global_context = GlobalContextPyramid(
            4, channels, global_token_dim, global_depth, global_heads, bottleneck_shape
        )
        self.input = nn.Conv3d(4, c0, 3, padding=1)
        self.enc0 = nn.ModuleList([TimeResBlock3D(c0, c0, time_dim, 8, 0.0) for _ in range(2)])
        self.down1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.enc1 = nn.ModuleList([TimeResBlock3D(c1, c1, time_dim, 8, 0.0) for _ in range(2)])
        self.down2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([TimeResBlock3D(c2, c2, time_dim, 8, 0.0) for _ in range(2)])
        self.down3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.mid = nn.ModuleList([TimeResBlock3D(c3, c3, time_dim, 8, 0.0) for _ in range(2)])
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList(
            [TimeResBlock3D(c2 * 2, c2, time_dim, 8, 0.0), TimeResBlock3D(c2, c2, time_dim, 8, 0.0)]
        )
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList(
            [TimeResBlock3D(c1 * 2, c1, time_dim, 8, 0.0), TimeResBlock3D(c1, c1, time_dim, 8, 0.0)]
        )
        self.up0 = nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1)
        self.dec0 = nn.ModuleList(
            [TimeResBlock3D(c0 * 2, c0, time_dim, 8, 0.0), TimeResBlock3D(c0, c0, time_dim, 8, 0.0)]
        )
        self.output = nn.Sequential(nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv3d(c0, 4, 3, padding=1))
        local_channels = (c0, c1, c2, c3, c2, c1, c0)
        source_channels = tuple(mri_channels[i] for i in self.MRI_LEVELS)
        self.zero_conv = nn.ModuleDict(
            {
                name: nn.Conv3d(src, dst, 1)
                for name, src, dst in zip(self.STAGES, source_channels, local_channels)
            }
        )
        for projection in self.zero_conv.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.bica = nn.ModuleDict(
            {
                name: TimeGatedBiCA(ch, token_dim, time_dim)
                for name, ch in zip(self.STAGES, local_channels)
            }
        )
        self.global_gate = nn.ModuleDict(
            {name: GlobalResidualGate(ch) for name, ch in zip(self.STAGES, local_channels)}
        )

    @staticmethod
    def _blocks(blocks, feature, time_embedding):
        for block in blocks:
            feature = block(feature, time_embedding)
        return feature

    def forward(self, z_k, k_normalized, mri, table, table_mask):
        if z_k.ndim != 5 or z_k.shape[1] != 4:
            raise ValueError("Bridge state must be [B,4,D,H,W]")
        time_embedding = self.time(k_normalized)
        structural = self.mri_encoder(mri) if self.use_mri_structure else None
        global_features = self.global_context(z_k) if self.use_global else None
        if self.use_clinical:
            table_state = self.table_encoder(table, table_mask)
            tokens = table_state["tokens"]
            mask = table_state["key_padding_mask"]
            observed = table_state["effective_observed"]
            reliability = table_state["reliability"]
        else:
            tokens = mask = observed = reliability = None

        def inject(name, feature, level):
            if structural is not None:
                source = structural[level]
                if source.shape[-3:] != feature.shape[-3:]:
                    source = F.interpolate(
                        source, size=feature.shape[-3:], mode="trilinear", align_corners=False
                    )
                feature = feature + self.zero_conv[name](source)
            nonlocal tokens
            if tokens is not None:
                feature, tokens = self.bica[name](
                    feature, tokens, mask, observed, reliability, time_embedding
                )
            return (
                self.global_gate[name](feature, global_features[level])
                if global_features is not None
                else feature
            )

        s0 = inject("enc0", self._blocks(self.enc0, self.input(z_k), time_embedding), 0)
        s1 = inject("enc1", self._blocks(self.enc1, self.down1(s0), time_embedding), 1)
        s2 = inject("enc2", self._blocks(self.enc2, self.down2(s1), time_embedding), 2)
        h = inject("mid", self._blocks(self.mid, self.down3(s2), time_embedding), 3)
        h = inject(
            "dec2", self._blocks(self.dec2, torch.cat([self.up2(h), s2], dim=1), time_embedding), 2
        )
        h = inject(
            "dec1", self._blocks(self.dec1, torch.cat([self.up1(h), s1], dim=1), time_embedding), 1
        )
        h = inject(
            "dec0", self._blocks(self.dec0, torch.cat([self.up0(h), s0], dim=1), time_embedding), 0
        )
        return self.output(h)
