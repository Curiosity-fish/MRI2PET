from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.paired_t1_fdg_dataset import PairedT1FDGDataset
from models.exp026.utils.patch_codec import build_patch_grid, extract_patches, stitch_patches
from models.exp032.data import load_labelmap, resolve_path
from models.exp032.evaluation import parc98_roi_metrics, patch_boundary_metrics
from models.exp032e.utils import ae_decode, ae_encode, collect_latent_stats, denormalize_latent, load_autoencoder, normalize_latent
from models.exp032f.vp_diffusion import LatentVPDiffusion
from metrics.image_metrics import compute_image_metrics_detailed
from models.exp032g_mri_control_unet import Exp032GMRIControlUNet3D, MRILatentProjectionHead3D


def parse_args():
    p = argparse.ArgumentParser(description="Exp032g MRI control strategy benchmark")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_csv(path: Path, header: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


class Exp032GMriDataset(Dataset):
    def __init__(self, base: PairedT1FDGDataset, parc98: torch.Tensor) -> None:
        self.base = base
        self.parc98 = parc98.float().contiguous()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        return {
            "mri": sample["input"],
            "pet": sample["target"],
            "brain_mask": sample["brain_mask"],
            "parc98": self.parc98,
            "subject_id": sample["subject_id"],
            "sample_id": sample["sample_id"],
        }


def build_dataset(config: dict, split: str) -> Exp032GMriDataset:
    data = config["data"]
    target_shape = tuple(int(v) for v in data["image_size"])
    base = PairedT1FDGDataset(
        pairs_csv=resolve_path(REPO_ROOT, data["pairs_csv"]),
        split_csv=resolve_path(REPO_ROOT, data["split_csv"]),
        split=split,
        fold=data.get("fold"),
        spatial_size=target_shape,
        max_samples=data.get(f"{split}_max_samples"),
        normalize_inputs=bool(data.get("normalize_inputs", False)),
        normalize_targets=bool(data.get("normalize_targets", False)),
        cache_dir=data.get("cache_dir"),
    )
    parc98 = load_labelmap(resolve_path(REPO_ROOT, data["parc98_labelmap_path"]), target_shape)
    return Exp032GMriDataset(base, parc98)


def build_loader(config: dict, split: str, shuffle: bool) -> DataLoader:
    ds = build_dataset(config, split)
    return DataLoader(
        ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", False)),
    )


def build_val_loader(config: dict) -> DataLoader:
    splits = list(config.get("evaluation", {}).get("val_splits", []))
    datasets = [build_dataset(config, s) for s in splits]
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", False)),
    )


def build_autoencoder(config: dict, device: torch.device):
    return load_autoencoder(str(config["ae"].get("kind", "ae")), REPO_ROOT / config["ae"]["checkpoint_path"], device)


def build_model(config: dict, latent_channels: int) -> Exp032GMRIControlUNet3D:
    m = config["model"]
    c = config["control"]
    return Exp032GMRIControlUNet3D(
        in_channels=latent_channels * 2,
        out_channels=latent_channels,
        base_channels=int(m["base_channels"]),
        channel_mults=tuple(m.get("channel_mults", [1, 2, 4])),
        dropout=float(m.get("dropout", 0.0)),
        num_groups=int(m.get("norm_groups", 8)),
        use_attention=bool(m.get("use_attention", False)),
        attention_level=int(m.get("attention_level", 1)),
        num_heads=int(m.get("num_heads", 4)),
        mri_base_channels=int(c.get("mri_base_channels", 8)),
        control_type=str(c["type"]),
        gamma_scale=float(c.get("gamma_scale", 0.5)),
        beta_scale=float(c.get("beta_scale", 0.5)),
    )


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def latent_from_pet(ae_model, pet: torch.Tensor, image_patch_coords, patch_size, latent_patch_coords, kind: str, patch_batch_size: int) -> torch.Tensor:
    patches = extract_patches(pet, image_patch_coords, patch_size)
    lat_chunks = []
    for start in range(0, len(latent_patch_coords), int(patch_batch_size)):
        end = min(start + int(patch_batch_size), len(latent_patch_coords))
        chunk = patches[:, start:end].reshape(-1, 1, *patch_size)
        latent, _ = ae_encode(ae_model, chunk, kind)
        lat_chunks.append(latent)
    lat_patches = torch.cat(
        [c.view(pet.shape[0], -1, ae_model.latent_channels, *tuple(int(v) // 4 for v in patch_size)) for c in lat_chunks],
        dim=1,
    )
    latent_size = tuple(int(v) // 4 for v in pet.shape[-3:])
    return stitch_patches(lat_patches, latent_patch_coords, latent_size)


def split_latent(latent_full: torch.Tensor, latent_patch_coords, latent_patch_size):
    return extract_patches(latent_full, latent_patch_coords, latent_patch_size)


def boundary_band_mae(pred: torch.Tensor, true: torch.Tensor, patch_size: tuple[int, int, int], band: int = 1) -> float:
    mask = torch.zeros_like(pred, dtype=torch.float32)
    for axis, width in enumerate(patch_size):
        dim = 2 + axis
        for boundary in range(int(width), pred.shape[dim], int(width)):
            lo = max(0, boundary - band)
            hi = min(pred.shape[dim], boundary + band + 1)
            slices = [slice(None)] * pred.ndim
            slices[dim] = slice(lo, hi)
            mask[tuple(slices)] = 1.0
    denom = mask.sum().clamp_min(1.0)
    return float((torch.abs(pred - true) * mask).sum().div(denom).item())


def local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size):
    pet = batch["pet"].to(device).float()
    mri = batch["mri"].to(device).float()
    ae_kind = str(cfg["ae"].get("kind", "ae")).lower()
    patch_batch_size = int(cfg["training"].get("patch_batch_size", 8))
    latent_full = latent_from_pet(ae_model, pet, image_patch_coords, patch_size, latent_patch_coords, ae_kind, patch_batch_size)
    latent_norm = normalize_latent(latent_full, stats)
    z0 = split_latent(latent_norm, latent_patch_coords, latent_patch_size).reshape(-1, latent_full.shape[1], *latent_patch_size)
    mri_patch = extract_patches(mri, image_patch_coords, patch_size).reshape(-1, 1, *patch_size)
    return pet, mri, z0, latent_norm, mri_patch


def decode_local(ae_model, z0_hat, pet_shape, latent_channels, latent_patch_coords, latent_patch_size, latent_full_size, stats):
    z0_full = stitch_patches(z0_hat.view(pet_shape[0], len(latent_patch_coords), latent_channels, *latent_patch_size), latent_patch_coords, latent_full_size)
    pred_latent = denormalize_latent(z0_full, stats)
    return ae_decode(ae_model, pred_latent), z0_full


def save_model(path: Path, model, optimizer, epoch: int, cfg: dict, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "payload": payload,
        },
        path,
    )


def ensure_initial_states(cfg: dict, model: Exp032GMRIControlUNet3D) -> tuple[Path, Path]:
    local_path = REPO_ROOT / cfg["experiment"]["initial_unet_state_path"]
    mri_path = REPO_ROOT / cfg["experiment"]["initial_mri_encoder_state_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    mri_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        torch.save({"local_state_dict": model.local_state_dict()}, local_path)
    if not mri_path.exists():
        torch.save({"mri_encoder_state_dict": model.mri_encoder_state_dict()}, mri_path)
    return local_path, mri_path


def load_initial_states(cfg: dict, model: Exp032GMRIControlUNet3D, device: torch.device) -> None:
    local_path, mri_path = ensure_initial_states(cfg, model)
    model.load_local_state_dict(torch.load(local_path, map_location=device)["local_state_dict"])
    model.load_mri_encoder_state_dict(torch.load(mri_path, map_location=device)["mri_encoder_state_dict"])


def pretrain_mri_encoder(cfg, model, ae_model, stats, train_loader, device, coords):
    if not bool(cfg.get("mri_pretrain", {}).get("enabled", False)):
        return None
    epochs = int(cfg["mri_pretrain"].get("epochs", 5))
    out_dir = REPO_ROOT / cfg["experiment"]["output_dir"]
    image_patch_coords, patch_size, latent_patch_coords, latent_patch_size = coords
    proj = MRILatentProjectionHead3D(model.mri_encoder, latent_channels=ae_model.latent_channels).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=float(cfg["mri_pretrain"].get("learning_rate", cfg["training"]["learning_rate"])), weight_decay=0.0)
    csv_path = out_dir / "logs" / "stage0_pretrain_log.csv"
    ensure_csv(csv_path, ["epoch", "align_loss", "cosine"])
    for epoch in range(1, epochs + 1):
        proj.train()
        loss_sum = 0.0
        cos_sum = 0.0
        seen = 0
        for batch in train_loader:
            _, _, z0, _, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
            pred = proj(mri_patch)
            loss = F.smooth_l1_loss(pred, z0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n = int(z0.shape[0])
            seen += n
            loss_sum += float(loss.item()) * n
            cos_sum += float(F.cosine_similarity(pred.flatten(1), z0.flatten(1), dim=1).mean().item()) * n
        append_csv(csv_path, [epoch, loss_sum / max(seen, 1), cos_sum / max(seen, 1)])
    stage0_path = out_dir / "checkpoints" / "stage0_mri_encoder.pt"
    torch.save({"mri_encoder_state_dict": model.mri_encoder_state_dict(), "epochs": epochs}, stage0_path)
    return stage0_path


def evaluate_epoch(model, ae_model, diffusion, loader, stats, cfg, device, coords, eval_modes=("normal",)):
    image_patch_coords, patch_size, latent_patch_coords, latent_patch_size = coords
    image_size = tuple(cfg["data"]["image_size"])
    latent_full_size = tuple(int(v) // 4 for v in image_size)
    rows = []
    bucket_losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pet, mri, z0, latent_norm, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
            cond = torch.zeros_like(z0)
            noise_ref = torch.randn_like(z0)
            for frac in [0.125, 0.375, 0.625, 0.875]:
                t = torch.full((z0.shape[0],), max(0, min(diffusion.timesteps - 1, int(round(frac * (diffusion.timesteps - 1))))), device=device, dtype=torch.long)
                zt, v_target = diffusion.q_sample(z0, t, noise=noise_ref)
                v_pred = model(zt, cond, t, mri_patch, control_enabled=True)
                bucket_losses.append(float(F.mse_loss(v_pred, v_target).item()))
            for mode in eval_modes:
                cur_mri = mri_patch
                control_enabled = True
                if mode == "control_off":
                    control_enabled = False
                elif mode == "shuffled_mri":
                    cur_mri = torch.roll(mri_patch.view(pet.shape[0], len(latent_patch_coords), 1, *patch_size), shifts=1, dims=0).reshape_as(mri_patch)
                for frac in [0.10, 0.25, 0.50, 0.75]:
                    t = torch.full((z0.shape[0],), max(0, min(diffusion.timesteps - 1, int(round(frac * (diffusion.timesteps - 1))))), device=device, dtype=torch.long)
                    zt, v_target = diffusion.q_sample(z0, t, noise=noise_ref)
                    v_pred = model(zt, cond, t, cur_mri, control_enabled=control_enabled)
                    z0_hat = diffusion.predict_z0_from_v(zt, t, v_pred)
                    recon, z0_hat_full = decode_local(ae_model, z0_hat, pet.shape, ae_model.latent_channels, latent_patch_coords, latent_patch_size, latent_full_size, stats)
                    metrics = compute_image_metrics_detailed(recon, pet, mask=batch["brain_mask"].to(device).float(), data_range=2.0)
                    roi = parc98_roi_metrics(recon, pet, batch["parc98"].to(device))
                    boundary = patch_boundary_metrics(recon, pet, patch_size)
                    band = boundary_band_mae(recon, pet, patch_size, band=int(cfg["evaluation"].get("boundary_band", 1)))
                    rows.append({
                        "eval_mode": mode,
                        "t_frac": frac,
                        "v_loss": float(F.mse_loss(v_pred, v_target).item()),
                        "latent_mae": float(torch.abs(z0_hat - z0).mean().item()),
                        "latent_rmse": float(torch.sqrt(torch.mean((z0_hat - z0).square())).item()),
                        "latent_cosine": float(F.cosine_similarity(z0_hat.flatten(1), z0.flatten(1), dim=1).mean().item()),
                        **{k: float(metrics[k]) for k in ["ssim", "psnr", "mae", "rmse", "brain_ssim", "brain_psnr", "brain_mae", "brain_rmse"]},
                        **roi,
                        **boundary,
                        "boundary_band_mae": band,
                    })
    return float(np.mean(bucket_losses)), rows


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = torch.device(cfg["training"].get("device", "cuda"))
    out_dir = REPO_ROOT / cfg["experiment"]["output_dir"]
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    image_size = tuple(cfg["data"]["image_size"])
    patch_size = tuple(cfg["data"]["patch_size"])
    image_patch_coords = build_patch_grid(image_size, patch_size)
    latent_patch_size = tuple(int(v) // 4 for v in patch_size)
    latent_full_size = tuple(int(v) // 4 for v in image_size)
    latent_patch_coords = build_patch_grid(latent_full_size, latent_patch_size)
    coords = (image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)

    ae_model = build_autoencoder(cfg, device)
    for p in ae_model.parameters():
        p.requires_grad_(False)
    model = build_model(cfg, ae_model.latent_channels).to(device)
    load_initial_states(cfg, model, device)
    diffusion = LatentVPDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_start=float(cfg["diffusion"].get("beta_start", 1e-4)),
        beta_end=float(cfg["diffusion"].get("beta_end", 2e-2)),
    ).to(device)
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_val_loader(cfg)

    stats_src = cfg["experiment"].get("latent_stats_path")
    stats_path = REPO_ROOT / stats_src if stats_src else out_dir / "latent_stats_ae.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
    else:
        stats = collect_latent_stats(ae_model, train_loader, "ae", device, image_patch_coords, patch_size, int(cfg["training"].get("patch_batch_size", 8)))
        torch.save(stats, stats_path)

    audit_batch = next(iter(train_loader))
    pet, mri, z0, latent_norm, mri_patch = local_inputs(audit_batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
    mri_features = model.mri_encoder(mri_patch)
    local_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("mri_encoder.") and not n.startswith("control."))
    mri_params = sum(p.numel() for p in model.mri_encoder.parameters())
    control_params = sum(p.numel() for p in model.control.parameters())
    audit = {
        "experiment": cfg["experiment"]["name"],
        "ae_checkpoint": str((REPO_ROOT / cfg["ae"]["checkpoint_path"]).resolve()),
        "latent_stats_path": str(stats_path.resolve()),
        "latent_patch_shape": list(z0.shape),
        "mri_patch_shape": list(mri_patch.shape),
        "mri_feature_shapes": {k: list(v.shape) for k, v in mri_features.items()},
        "local_unet_params": int(local_params),
        "mri_encoder_params": int(mri_params),
        "control_params": int(control_params),
        "total_trainable_params": count_params(model)[1],
        "control_type": cfg["control"]["type"],
        "seed": int(cfg["experiment"].get("seed", 42)),
        "epochs": int(cfg["training"]["epochs"]),
        "timesteps": int(diffusion.timesteps),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
    print(json.dumps(audit, indent=2))

    if bool(cfg.get("mri_pretrain", {}).get("enabled", False)):
        pretrain_mri_encoder(cfg, model, ae_model, stats, train_loader, device, coords)
        # Stage1 starts from the same Local UNet init and zero-init control; only MRI encoder is pretrained.
        pretrained_mri = model.mri_encoder_state_dict()
        model = build_model(cfg, ae_model.latent_channels).to(device)
        load_initial_states(cfg, model, device)
        model.load_mri_encoder_state_dict(pretrained_mri)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["epochs"]))

    if args.smoke_test:
        model.train()
        cond = torch.zeros_like(z0)
        t = torch.randint(0, diffusion.timesteps - 1, (z0.shape[0],), device=device)
        zt, v_target = diffusion.q_sample(z0, t, noise=torch.randn_like(z0))
        v_on = model(zt, cond, t, mri_patch, control_enabled=True)
        v_off = model(zt, cond, t, mri_patch, control_enabled=False)
        loss = F.mse_loss(v_on, v_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        z0_hat = diffusion.predict_z0_from_v(zt, t, v_on)
        recon, _ = decode_local(ae_model, z0_hat, pet.shape, ae_model.latent_channels, latent_patch_coords, latent_patch_size, latent_full_size, stats)
        tmp_csv = out_dir / "logs" / "smoke_train_log.csv"
        ensure_csv(tmp_csv, ["ok", "loss"])
        append_csv(tmp_csv, [1, float(loss.item())])
        smoke = {
            "v_loss": float(loss.item()),
            "v_pred_shape": list(v_on.shape),
            "recon_shape": list(recon.shape),
            "initial_control_on_off_mae": float(torch.abs(v_on.detach() - v_off.detach()).mean().item()),
            "ae_grad_count": sum(1 for p in ae_model.parameters() if p.grad is not None),
            "mri_encoder_grad_count": sum(1 for p in model.mri_encoder.parameters() if p.grad is not None),
            "control_grad_count": sum(1 for p in model.control.parameters() if p.grad is not None),
            "control_stats": getattr(model.control, "last_stats", {}),
        }
        print(json.dumps(smoke, indent=2))
        return

    train_csv = out_dir / "logs" / "train_log.csv"
    val_csv = out_dir / "logs" / "val_log.csv"
    ensure_csv(train_csv, ["epoch", "train_v_loss", "val_v_loss", "lr", "control_weight_norm", "epoch_time_sec"])
    ensure_csv(val_csv, ["epoch", "eval_mode", "t_frac", "v_loss", "latent_mae", "latent_rmse", "latent_cosine", "ssim", "psnr", "mae", "rmse", "brain_ssim", "brain_psnr", "brain_mae", "brain_rmse", "roi_mae", "roi_rmse", "roi_pearson", "roi_spearman", "boundary_mae", "boundary_gradient_error", "boundary_band_mae"])
    best = float("inf")
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        start_time = time.time()
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            pet, _, z0, _, mri_patch = local_inputs(batch, ae_model, stats, cfg, device, image_patch_coords, patch_size, latent_patch_coords, latent_patch_size)
            t = torch.randint(0, diffusion.timesteps - 1, (z0.shape[0],), device=device)
            zt, v_target = diffusion.q_sample(z0, t, noise=torch.randn_like(z0))
            v_pred = model(zt, torch.zeros_like(z0), t, mri_patch, control_enabled=True)
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += int(z0.shape[0])
            loss_sum += float(loss.item()) * int(z0.shape[0])
        scheduler.step()
        train_loss = loss_sum / max(seen, 1)
        eval_modes = ("normal",) if epoch != int(cfg["training"]["epochs"]) else ("normal", "control_off", "shuffled_mri")
        val_loss, val_rows = evaluate_epoch(model, ae_model, diffusion, val_loader, stats, cfg, device, coords, eval_modes=eval_modes)
        append_csv(train_csv, [epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"], getattr(model.control, "control_weight_norm", lambda: 0.0)(), time.time() - start_time])
        df = pd.DataFrame([{**{"epoch": epoch}, **r} for r in val_rows])
        df.to_csv(val_csv, mode="a", header=False, index=False)
        payload = {"epoch": epoch, "train_v_loss": train_loss, "val_v_loss": val_loss, "audit": audit}
        save_model(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, cfg, payload)
        if val_loss < best:
            best = val_loss
            save_model(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, cfg, payload)


if __name__ == "__main__":
    main()
