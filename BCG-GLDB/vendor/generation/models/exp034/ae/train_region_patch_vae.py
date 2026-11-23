from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.paired_t1_fdg_dataset import PairedT1FDGDataset
from engine.trainer_t1_fdg import append_log, prepare_experiment_dir
from losses.diffusion_losses import low_frequency_l1, masked_l1
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp025d_global_region_patch_denoiser import _region_boundary
from models.exp025p_region_patch_latent import Exp025PRegionPatchLatentVAE3D
from utils.patch_latent_codec import build_patch_grid, extract_patches, stitch_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp025p stage1 region-aware patch latent VAE.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["experiment"]["config_path"] = str(path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(config: dict, split: str) -> DataLoader:
    dataset = PairedT1FDGDataset(
        pairs_csv=REPO_ROOT / config["data"]["pairs_csv"],
        split_csv=REPO_ROOT / config["data"]["split_csv"],
        split=split,
        fold=None,
        spatial_size=tuple(config["data"]["image_size"]),
        max_samples=config["data"].get(f"{split}_max_samples"),
        cache_dir=(REPO_ROOT / config["data"]["cache_dir"]) if config["data"].get("cache_dir") else None,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        shuffle=(split == "train"),
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        drop_last=(split == "train"),
    )


def ensure_csv_header(path: Path, header: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(header)
        return
    first_line = path.read_text(encoding="utf-8").splitlines()[:1]
    if first_line != [",".join(header)]:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(header)


def append_csv_row(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def load_labelmap(path: Path, image_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    labels = torch.from_numpy(nib.load(str(path)).get_fdata(dtype="float32")).long()[None, None]
    if tuple(labels.shape[-3:]) != tuple(image_size):
        labels = F.interpolate(labels.float(), size=image_size, mode="nearest").long()
    return labels.to(device=device)


def region_features(labels: torch.Tensor, brain_mask: torch.Tensor, num_regions: int) -> torch.Tensor:
    label_norm = labels.float() / float(max(int(num_regions), 1))
    boundary = _region_boundary(labels)
    return torch.cat([brain_mask.float(), label_norm, boundary], dim=1)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def region_mean_l1(pred: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, brain_mask: torch.Tensor, num_regions: int) -> torch.Tensor:
    diff = (pred - target).abs()
    terms = []
    for region_id in range(1, int(num_regions) + 1):
        mask = ((labels == region_id) & (brain_mask > 0.5)).to(dtype=diff.dtype)
        denom = mask.sum()
        if denom > 0:
            terms.append((diff * mask).sum() / denom.clamp_min(1.0))
    if not terms:
        return diff.new_zeros(())
    return torch.stack(terms).mean()


def infer_patch_latent_shape(
    model: Exp025PRegionPatchLatentVAE3D,
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> tuple[int, int, int, int]:
    with torch.no_grad():
        dummy_pet = torch.zeros((1, 1, *patch_size), device=device)
        dummy_region = (
            torch.zeros((1, model.region_channels, *patch_size), device=device)
            if model.region_channels > 0
            else None
        )
        mu, _ = model.encode(dummy_pet, dummy_region)
    return tuple(int(v) for v in mu.shape[1:])


def reconstruct_full_pet(
    generator: Exp025PRegionPatchLatentVAE3D,
    pet: torch.Tensor,
    region_cond: torch.Tensor | None,
    patch_coords: list[tuple[int, int, int]],
    patch_size: tuple[int, int, int],
    patch_batch_size: int,
) -> torch.Tensor:
    pet_patches = extract_patches(pet, patch_coords, patch_size)
    region_patches = extract_patches(region_cond, patch_coords, patch_size) if region_cond is not None else None
    chunks = []
    for start in range(0, len(patch_coords), int(patch_batch_size)):
        end = min(start + int(patch_batch_size), len(patch_coords))
        pet_chunk = pet_patches[:, start:end].reshape(-1, 1, *patch_size)
        region_chunk = (
            region_patches[:, start:end].reshape(-1, region_cond.shape[1], *patch_size)
            if region_patches is not None and region_cond is not None
            else None
        )
        recon, _, _, _ = generator(pet_chunk, region_chunk)
        chunks.append(recon)
    recon_patches = torch.cat(chunks, dim=0).view(pet.shape[0], len(patch_coords), 1, *patch_size)
    return stitch_patches(recon_patches, patch_coords, tuple(pet.shape[-3:]))


def train_epoch(
    generator: Exp025PRegionPatchLatentVAE3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    labels: torch.Tensor | None,
    patch_coords: list[tuple[int, int, int]],
    patch_size: tuple[int, int, int],
) -> dict[str, float]:
    generator.train()
    sums = {key: 0.0 for key in ["loss_total", "loss_patch", "loss_kl", "loss_full", "loss_region", "loss_boundary", "loss_freq"]}
    seen = 0
    patch_count = len(patch_coords)
    patch_batch_size = int(config["training"].get("patch_batch_size", patch_count))
    num_regions = int(config["data"].get("num_regions", 98))
    beta = float(config["loss"].get("beta", 1e-3))
    lambda_patch = float(config["loss"].get("lambda_patch_recon", 1.0))
    lambda_full = float(config["loss"].get("lambda_full_recon", 1.0))
    lambda_region = float(config["loss"].get("lambda_region_mean", 0.2))
    lambda_boundary = float(config["loss"].get("lambda_boundary", 0.2))
    lambda_freq = float(config["loss"].get("lambda_full_freq", 0.0))

    for batch in tqdm(loader, desc="exp025p stage1 train", leave=False):
        pet = batch["target"].to(device)
        brain_mask = batch["brain_mask"].to(device)
        batch_labels = labels.expand(pet.shape[0], -1, -1, -1, -1) if labels is not None else None
        region_cond = (
            region_features(batch_labels, brain_mask, num_regions)
            if generator.region_channels > 0 and batch_labels is not None
            else None
        )
        pet_patches = extract_patches(pet, patch_coords, patch_size)
        brain_patches = extract_patches(brain_mask, patch_coords, patch_size)
        region_patches = extract_patches(region_cond, patch_coords, patch_size) if region_cond is not None else None

        optimizer.zero_grad(set_to_none=True)
        patch_terms = []
        kl_terms = []
        recon_chunks = []
        for start in range(0, patch_count, patch_batch_size):
            end = min(start + patch_batch_size, patch_count)
            pet_chunk = pet_patches[:, start:end].reshape(-1, 1, *patch_size)
            mask_chunk = brain_patches[:, start:end].reshape(-1, 1, *patch_size)
            region_chunk = (
                region_patches[:, start:end].reshape(-1, region_cond.shape[1], *patch_size)
                if region_patches is not None and region_cond is not None
                else None
            )
            recon, mu, logvar, _ = generator(pet_chunk, region_chunk)
            weight = float(end - start)
            patch_terms.append(masked_l1(recon, pet_chunk, mask_chunk) * weight)
            kl_terms.append(kl_divergence(mu, logvar) * weight)
            recon_chunks.append(recon)

        recon_patches = torch.cat(recon_chunks, dim=0).view(pet.shape[0], patch_count, 1, *patch_size)
        full_recon = stitch_patches(recon_patches, patch_coords, tuple(pet.shape[-3:]))
        loss_patch = torch.stack(patch_terms).sum() / float(max(patch_count, 1))
        loss_kl = torch.stack(kl_terms).sum() / float(max(patch_count, 1))
        loss_full = masked_l1(full_recon, pet, brain_mask)
        loss_region = (
            region_mean_l1(full_recon, pet, batch_labels, brain_mask, num_regions)
            if lambda_region > 0 and batch_labels is not None
            else pet.new_zeros(())
        )
        loss_boundary = (
            masked_l1(full_recon, pet, region_cond[:, 2:3] * brain_mask)
            if lambda_boundary > 0 and region_cond is not None
            else pet.new_zeros(())
        )
        loss_freq = low_frequency_l1(full_recon * brain_mask, pet * brain_mask) if lambda_freq > 0 else pet.new_zeros(())
        loss_total = (
            lambda_patch * loss_patch
            + beta * loss_kl
            + lambda_full * loss_full
            + lambda_region * loss_region
            + lambda_boundary * loss_boundary
            + lambda_freq * loss_freq
        )
        loss_total.backward()
        optimizer.step()

        bs = int(pet.shape[0])
        seen += bs
        for key, value in [
            ("loss_total", loss_total),
            ("loss_patch", loss_patch),
            ("loss_kl", loss_kl),
            ("loss_full", loss_full),
            ("loss_region", loss_region),
            ("loss_boundary", loss_boundary),
            ("loss_freq", loss_freq),
        ]:
            sums[key] += float(value.item()) * bs
    return {key: value / max(seen, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate(
    generator: Exp025PRegionPatchLatentVAE3D,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    labels: torch.Tensor | None,
    patch_coords: list[tuple[int, int, int]],
    patch_size: tuple[int, int, int],
) -> dict[str, float]:
    generator.eval()
    rows = []
    num_regions = int(config["data"].get("num_regions", 98))
    for batch in loader:
        pet = batch["target"].to(device)
        brain_mask = batch["brain_mask"].to(device)
        batch_labels = labels.expand(pet.shape[0], -1, -1, -1, -1) if labels is not None else None
        region_cond = (
            region_features(batch_labels, brain_mask, num_regions)
            if generator.region_channels > 0 and batch_labels is not None
            else None
        )
        recon = reconstruct_full_pet(
            generator,
            pet,
            region_cond,
            patch_coords,
            patch_size,
            int(config["training"].get("patch_batch_size", len(patch_coords))),
        )
        metrics = compute_image_metrics_detailed(recon, pet, mask=brain_mask, data_range=2.0)
        rows.append({k: float(metrics[k]) for k in ["mae", "rmse", "psnr", "ssim"]})
    frame = pd.DataFrame(rows)
    return {key: float(frame[key].mean()) for key in ["mae", "rmse", "psnr", "ssim"]}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device(config["training"].get("device", "cuda"))
    paths = prepare_experiment_dir(REPO_ROOT / config["experiment"]["output_dir"], config)

    image_size = tuple(int(v) for v in config["data"]["image_size"])
    patch_size = tuple(int(v) for v in config["data"]["patch_size"])
    patch_coords = build_patch_grid(image_size, patch_size)
    model_cfg = config.get("model", {})
    region_channels = int(model_cfg.get("region_channels", 0))
    use_region_labels = (
        region_channels > 0
        or float(config["loss"].get("lambda_region_mean", 0.0)) > 0
        or float(config["loss"].get("lambda_boundary", 0.0)) > 0
    )
    labels = (
        load_labelmap(REPO_ROOT / config["data"]["region_mask_path"], image_size, device)
        if use_region_labels
        else None
    )

    train_loader = build_loader(config, "train")
    val_splits = list(config["evaluation"].get("val_splits", []))
    val_loaders = {split: build_loader(config, split) for split in val_splits}

    generator = Exp025PRegionPatchLatentVAE3D(
        image_channels=1,
        region_channels=region_channels,
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        norm_groups=int(model_cfg.get("norm_groups", 8)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    patch_latent_shape = infer_patch_latent_shape(generator, patch_size, device)

    train_csv = paths.root / "train_log.csv"
    val_csv = paths.root / "val_log.csv"
    for csv_path in [train_csv, val_csv]:
        if csv_path.exists():
            csv_path.unlink()
    ensure_csv_header(train_csv, ["epoch", "train_total", "train_patch", "train_kl", "train_full", "train_region", "train_boundary", "train_freq"])
    ensure_csv_header(val_csv, ["epoch", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"])

    best_ssim = float("-inf")
    epochs = int(config["training"].get("epochs", 100))
    sample_interval = int(config["evaluation"].get("sample_val_interval", 1))
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(generator, train_loader, optimizer, device, config, labels, patch_coords, patch_size)
        append_csv_row(
            train_csv,
            [
                epoch,
                train_metrics["loss_total"],
                train_metrics["loss_patch"],
                train_metrics["loss_kl"],
                train_metrics["loss_full"],
                train_metrics["loss_region"],
                train_metrics["loss_boundary"],
                train_metrics["loss_freq"],
            ],
        )
        log_row = {"epoch": epoch, **train_metrics}
        if sample_interval > 0 and epoch % sample_interval == 0:
            fold_rows = []
            for split, loader in val_loaders.items():
                metrics = evaluate(generator, loader, device, config, labels, patch_coords, patch_size)
                metrics["val_split"] = split
                fold_rows.append(metrics)
            pd.DataFrame(fold_rows).to_csv(paths.predictions / f"sampling_epoch_{epoch:04d}_fold_summary.csv", index=False)
            frame = pd.DataFrame(fold_rows)
            agg = {
                "mae_mean": float(frame["mae"].mean()),
                "mae_std": float(frame["mae"].std(ddof=0)),
                "rmse_mean": float(frame["rmse"].mean()),
                "rmse_std": float(frame["rmse"].std(ddof=0)),
                "psnr_mean": float(frame["psnr"].mean()),
                "psnr_std": float(frame["psnr"].std(ddof=0)),
                "ssim_mean": float(frame["ssim"].mean()),
                "ssim_std": float(frame["ssim"].std(ddof=0)),
            }
            append_csv_row(val_csv, [epoch, agg["mae_mean"], agg["mae_std"], agg["rmse_mean"], agg["rmse_std"], agg["psnr_mean"], agg["psnr_std"], agg["ssim_mean"], agg["ssim_std"]])
            log_row.update(agg)
            checkpoint = {
                "generator_state_dict": generator.state_dict(),
                "epoch": epoch,
                "patch_size": patch_size,
                "patch_latent_shape": patch_latent_shape,
                "config": config,
            }
            torch.save(checkpoint, paths.checkpoints / "last.pt")
            if agg["ssim_mean"] > best_ssim:
                best_ssim = agg["ssim_mean"]
                torch.save(checkpoint, paths.checkpoints / "best.pt")
        append_log(paths.train_log, str(log_row))


if __name__ == "__main__":
    main()
