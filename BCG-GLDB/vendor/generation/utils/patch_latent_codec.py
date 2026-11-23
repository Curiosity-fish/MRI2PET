from __future__ import annotations

from typing import Iterable

import torch


def build_patch_grid(
    image_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    coords: list[tuple[int, int, int]] = []
    for z in range(0, int(image_size[0]), int(patch_size[0])):
        for y in range(0, int(image_size[1]), int(patch_size[1])):
            for x in range(0, int(image_size[2]), int(patch_size[2])):
                coords.append((z, y, x))
    return coords


def extract_patches(
    volume: torch.Tensor,
    coords: Iterable[tuple[int, int, int]],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    pd, ph, pw = [int(v) for v in patch_size]
    patches = [
        volume[..., z : z + pd, y : y + ph, x : x + pw]
        for z, y, x in coords
    ]
    return torch.stack(patches, dim=1)


def stitch_patches(
    patches: torch.Tensor,
    coords: Iterable[tuple[int, int, int]],
    output_size: tuple[int, int, int],
) -> torch.Tensor:
    output = torch.zeros(
        (patches.shape[0], patches.shape[2], *output_size),
        device=patches.device,
        dtype=patches.dtype,
    )
    for patch_idx, (z, y, x) in enumerate(coords):
        patch = patches[:, patch_idx]
        output[..., z : z + patch.shape[-3], y : y + patch.shape[-2], x : x + patch.shape[-1]] = patch
    return output


def encode_patch_latents(
    dvae: torch.nn.Module,
    volume: torch.Tensor,
    patch_coords: list[tuple[int, int, int]],
    patch_size: tuple[int, int, int],
    patch_batch_size: int,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    patches = extract_patches(volume, patch_coords, patch_size)
    latent_chunks = []
    latent_spatial: tuple[int, int, int] | None = None
    for start in range(0, len(patch_coords), int(patch_batch_size)):
        end = min(start + int(patch_batch_size), len(patch_coords))
        patch_chunk = patches[:, start:end].reshape(-1, 1, *patch_size)
        with torch.no_grad():
            mu, _ = dvae.encoder(patch_chunk)
        latent_chunks.append(mu)
        if latent_spatial is None:
            latent_spatial = tuple(int(v) for v in mu.shape[-3:])
    latent_patches = torch.cat(latent_chunks, dim=0).view(
        volume.shape[0],
        len(patch_coords),
        -1,
        *latent_spatial,
    )
    assert latent_spatial is not None
    return latent_patches, latent_spatial


def decode_patch_latents(
    dvae: torch.nn.Module,
    latent_volume: torch.Tensor,
    latent_patch_coords: list[tuple[int, int, int]],
    latent_patch_size: tuple[int, int, int],
    image_patch_coords: list[tuple[int, int, int]],
    image_size: tuple[int, int, int],
    patch_batch_size: int,
) -> torch.Tensor:
    latent_patches = extract_patches(latent_volume, latent_patch_coords, latent_patch_size)
    decoded_chunks = []
    decoded_patch_size: tuple[int, int, int] | None = None
    for start in range(0, len(latent_patch_coords), int(patch_batch_size)):
        end = min(start + int(patch_batch_size), len(latent_patch_coords))
        latent_chunk = latent_patches[:, start:end].reshape(-1, latent_volume.shape[1], *latent_patch_size)
        with torch.no_grad():
            decoded = dvae.decoder(latent_chunk)
        decoded_chunks.append(decoded)
        if decoded_patch_size is None:
            decoded_patch_size = tuple(int(v) for v in decoded.shape[-3:])
    assert decoded_patch_size is not None
    decoded_patches = torch.cat(decoded_chunks, dim=0).view(
        latent_volume.shape[0],
        len(latent_patch_coords),
        1,
        *decoded_patch_size,
    )
    return stitch_patches(decoded_patches, image_patch_coords, image_size)
