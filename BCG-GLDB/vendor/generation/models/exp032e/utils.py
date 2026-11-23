from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from models.exp032.core import AE, KLVAE
from models.exp026.utils.patch_codec import build_patch_grid, extract_patches, stitch_patches


def load_autoencoder(kind: str, checkpoint_path: str | Path, device: torch.device):
    kind = kind.lower()
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    kwargs = dict(
        image_channels=1,
        latent_channels=int(model_cfg.get('latent_channels', 4)),
        base_channels=int(model_cfg.get('base_channels', 16)),
        norm_groups=int(model_cfg.get('norm_groups', 8)),
        dropout=float(model_cfg.get('dropout', 0.0)),
    )
    model = (AE(**kwargs) if kind == 'ae' else KLVAE(**kwargs)).to(device)
    state = ckpt.get('model_state_dict', ckpt.get('generator_state_dict', ckpt.get('model', ckpt)))
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def ae_encode(model, x: torch.Tensor, kind: str):
    if kind.lower() == 'klvae':
        mu, logvar = model.encode(x)
        return mu, {'mu': mu, 'logvar': logvar}
    z = model.encode(x)
    return z, {'z': z}


def ae_decode(model, z: torch.Tensor):
    return model.decoder(z)


def collect_latent_stats(model, loader, kind: str, device: torch.device, patch_coords, patch_size, patch_batch_size: int = 8):
    sum_ = None
    sumsq = None
    count = 0
    all_latents = []
    with torch.inference_mode():
        for batch in loader:
            pet = batch['pet'].to(device).float()
            patches = extract_patches(pet, patch_coords, patch_size)
            lat_chunks = []
            for start in range(0, len(patch_coords), int(patch_batch_size)):
                end = min(start + int(patch_batch_size), len(patch_coords))
                patch_chunk = patches[:, start:end].reshape(-1, 1, *patch_size)
                latent, _ = ae_encode(model, patch_chunk, kind)
                lat_chunks.append(latent)
            latent = torch.cat(lat_chunks, dim=0)
            flat = latent.permute(1,0,2,3,4).contiguous().view(latent.shape[1], -1)
            cur_sum = flat.sum(dim=1)
            cur_sumsq = (flat * flat).sum(dim=1)
            cur_count = flat.shape[1]
            sum_ = cur_sum if sum_ is None else sum_ + cur_sum
            sumsq = cur_sumsq if sumsq is None else sumsq + cur_sumsq
            count += cur_count
            all_latents.append(latent.detach().cpu())
    mean = sum_ / max(count, 1)
    var = (sumsq / max(count, 1)) - mean * mean
    std = torch.sqrt(var.clamp_min(1e-8))
    lat = torch.cat(all_latents, dim=0)
    inactive = (lat.flatten(2).std(dim=2, unbiased=False) < 1e-3).float().mean().item()
    return {'mean': mean, 'std': std, 'inactive_channel_ratio': float(inactive), 'latent_mean': float(lat.mean()), 'latent_std': float(lat.std(unbiased=False))}


def latent_grid_coords(image_size, patch_size):
    return build_patch_grid(tuple(int(v)//4 for v in image_size), tuple(int(v)//4 for v in patch_size))

def normalize_latent(z, stats):
    mean = stats['mean'].to(z.device).view(1, -1, 1, 1, 1)
    std = stats['std'].to(z.device).view(1, -1, 1, 1, 1)
    return (z - mean) / (std + 1e-6)


def denormalize_latent(z, stats):
    mean = stats['mean'].to(z.device).view(1, -1, 1, 1, 1)
    std = stats['std'].to(z.device).view(1, -1, 1, 1, 1)
    return z * (std + 1e-6) + mean
