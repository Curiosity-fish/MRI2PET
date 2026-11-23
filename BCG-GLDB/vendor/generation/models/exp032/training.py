from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from exp026.utils.patch_codec import build_patch_grid, extract_patches, stitch_patches
from exp032.evaluation import parc98_roi_metrics, patch_boundary_metrics
from metrics.image_metrics import compute_image_metrics_detailed


@dataclass
class PatchAutoencoderOutputs:
    full_metrics: dict[str, float]
    patch_metrics: dict[str, float]
    roi_metrics: dict[str, float]
    boundary_metrics: dict[str, float]
    latent_stats: dict[str, float]


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def latent_statistics(mu: torch.Tensor, logvar: torch.Tensor | None = None) -> dict[str, float]:
    stats = {
        "latent_mean": float(mu.mean().item()),
        "latent_std": float(mu.std(unbiased=False).item()),
        "latent_min": float(mu.min().item()),
        "latent_max": float(mu.max().item()),
    }
    if logvar is not None:
        stats.update(
            {
                "mu_mean": float(mu.mean().item()),
                "mu_std": float(mu.std(unbiased=False).item()),
                "mu_min": float(mu.min().item()),
                "mu_max": float(mu.max().item()),
                "logvar_mean": float(logvar.mean().item()),
                "logvar_std": float(logvar.std(unbiased=False).item()),
                "logvar_min": float(logvar.min().item()),
                "logvar_max": float(logvar.max().item()),
                "posterior_std_mean": float(torch.exp(0.5 * logvar).mean().item()),
                "posterior_std_std": float(torch.exp(0.5 * logvar).std(unbiased=False).item()),
            }
        )
    return stats


def evaluate_patch_autoencoder(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    data_range: float,
    patch_batch_size: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
    model.eval()
    patch_coords = build_patch_grid(tuple(int(v) * 4 for v in patch_size), patch_size)
    full_rows = []
    patch_rows = []
    roi_rows = []
    boundary_rows = []
    with torch.inference_mode():
        for batch in loader:
            pet = batch["pet"].to(device).float()
            patches = extract_patches(pet, patch_coords, patch_size)
            recon_chunks, mu_chunks, logvar_chunks = [], [], []
            for start in range(0, len(patch_coords), patch_batch_size):
                end = min(start + patch_batch_size, len(patch_coords))
                patch_chunk = patches[:, start:end].reshape(-1, 1, *patch_size)
                recon, mu, logvar, _ = model(patch_chunk)
                recon_chunks.append(recon)
                mu_chunks.append(mu)
                logvar_chunks.append(logvar)
            recon_patches = torch.cat(recon_chunks, dim=0).view(pet.shape[0], len(patch_coords), 1, *patch_size)
            recon = stitch_patches(recon_patches, patch_coords, tuple(pet.shape[-3:]))
            for i in range(pet.shape[0]):
                pred = recon[i : i + 1]
                true = pet[i : i + 1]
                brain_mask = batch["brain_mask"][i : i + 1].to(device).float()
                metrics = compute_image_metrics_detailed(pred, true, mask=brain_mask, data_range=data_range)
                full_rows.append({k: float(metrics[k]) for k in ["mae", "rmse", "psnr", "ssim", "brain_mae", "brain_rmse", "brain_psnr", "brain_ssim"]})
                patch_metrics = compute_image_metrics_detailed(recon_patches[i].reshape(-1, 1, *patch_size), patches[i].reshape(-1, 1, *patch_size), mask=None, data_range=data_range)
                patch_rows.append({k: float(patch_metrics[k]) for k in ["mae", "rmse", "psnr", "ssim"]})
                roi_rows.append(parc98_roi_metrics(pred, true, batch["parc98"][i : i + 1].to(device)))
                boundary_rows.append(patch_boundary_metrics(pred, true, patch_size))
    return full_rows, patch_rows, roi_rows, boundary_rows

