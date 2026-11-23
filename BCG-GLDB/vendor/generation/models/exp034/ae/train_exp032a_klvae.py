
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp026.utils.checkpoints import save_checkpoint
from exp026.utils.patch_codec import build_patch_grid, extract_patches, stitch_patches
from exp032.data import build_dataset
from exp032.core import build_exp032_model, KLVAE, AE, VQVAE, WAE
from exp032.evaluation import parc98_roi_metrics, patch_boundary_metrics
from exp032.training import kl_divergence, latent_statistics
from metrics.image_metrics import compute_image_metrics_detailed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Exp032 patch autoencoder benchmarks")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["experiment"]["config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(config: dict, split: str) -> DataLoader:
    dataset = build_dataset(config, REPO_ROOT, split)
    return DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=(split == "train"), num_workers=int(config["training"].get("num_workers", 0)), pin_memory=bool(config["training"].get("pin_memory", False)))


def build_val_loader(config: dict) -> DataLoader:
    splits = list(config.get("evaluation", {}).get("val_splits", []))
    if not splits:
        raise ValueError("evaluation.val_splits must not be empty")
    datasets = [build_dataset(config, REPO_ROOT, s) for s in splits]
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(ds, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"].get("num_workers", 0)), pin_memory=bool(config["training"].get("pin_memory", False)))


def build_model(config: dict) -> torch.nn.Module:
    method = str(config["experiment"]["method"]).lower()
    model_cfg = config["model"]
    kwargs = dict(image_channels=1, latent_channels=int(model_cfg["latent_channels"]), base_channels=int(model_cfg["base_channels"]), norm_groups=int(model_cfg["norm_groups"]), dropout=float(model_cfg.get("dropout", 0.0)))
    if method == "vqvae":
        kwargs.update(codebook_size=int(model_cfg.get("codebook_size", 256)), commitment_weight=float(model_cfg.get("commitment_weight", 0.25)))
    return build_exp032_model(method, **kwargs)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def mmd_imq(z: torch.Tensor, prior: torch.Tensor, scales=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0)) -> torch.Tensor:
    z = z.flatten(1)
    prior = prior.flatten(1)
    def pdist2(a, b):
        a2 = (a.square()).sum(1, keepdim=True)
        b2 = (b.square()).sum(1, keepdim=True).t()
        return (a2 - 2 * a @ b.t() + b2).clamp_min(0.0)
    xx = pdist2(z, z)
    yy = pdist2(prior, prior)
    xy = pdist2(z, prior)
    out = 0.0
    dim = z.shape[1]
    for s in scales:
        c = 2.0 * dim * s
        out = out + (c / (c + xx)).mean() + (c / (c + yy)).mean() - 2.0 * (c / (c + xy)).mean()
    return out / float(len(scales))


def recon_from_model(model: torch.nn.Module, method: str, patch: torch.Tensor):
    if method == "klvae":
        recon, mu, logvar, z = model(patch)
        return recon, {"mu": mu, "logvar": logvar, "z": z}
    if method == "ae":
        recon, z = model(patch)
        return recon, {"z": z}
    if method == "vqvae":
        recon, z_e, z_q, indices, codebook, commit = model(patch)
        return recon, {"z_e": z_e, "z_q": z_q, "indices": indices, "codebook": codebook, "commit": commit}
    if method == "wae":
        recon, z = model(patch)
        return recon, {"z": z}
    raise ValueError(method)


def latent_tensor_from_payload(method: str, payload: dict[str, torch.Tensor]) -> torch.Tensor:
    if method == "klvae":
        return payload["mu"]
    if method == "vqvae":
        return payload["z_q"]
    return payload["z"]


def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: dict, method: str, patch_coords, patch_size, image_size, patch_batch_size: int, out_dir: Path):
    model.eval()
    rows = []
    patch_rows = []
    latents = []
    extras = []
    first_visual = None
    reconstruction_key = str(config["data"].get("reconstruction_key", "pet"))
    with torch.inference_mode():
        for batch in loader:
            pet = batch[reconstruction_key].to(device).float()
            patches = extract_patches(pet, patch_coords, patch_size)
            recon_chunks = []
            patch_metrics_src = []
            for start in range(0, len(patch_coords), patch_batch_size):
                end = min(start + patch_batch_size, len(patch_coords))
                patch_chunk = patches[:, start:end].reshape(-1, 1, *patch_size)
                recon, payload = recon_from_model(model, method, patch_chunk)
                recon_chunks.append(recon)
                latents.append(latent_tensor_from_payload(method, payload).detach().cpu())
                if method == "klvae":
                    extras.append({"mu": payload["mu"].detach().cpu(), "logvar": payload["logvar"].detach().cpu(), "z": payload["z"].detach().cpu()})
                elif method == "vqvae":
                    extras.append({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in payload.items()})
                else:
                    extras.append({"z": payload["z"].detach().cpu()})
                patch_metrics_src.append((recon, patch_chunk))
            recon_patches = torch.cat(recon_chunks, dim=0).view(pet.shape[0], len(patch_coords), 1, *patch_size)
            recon_full = stitch_patches(recon_patches, patch_coords, image_size)
            if first_visual is None:
                first_visual = (pet[0:1].detach().cpu(), recon_full[0:1].detach().cpu())
            for i in range(pet.shape[0]):
                pred = recon_full[i:i+1]
                true = pet[i:i+1]
                brain_mask = batch["brain_mask"][i:i+1].to(device).float()
                metrics = compute_image_metrics_detailed(pred, true, mask=brain_mask, data_range=2.0)
                roi = parc98_roi_metrics(pred, true, batch["parc98"][i:i+1].to(device))
                boundary = patch_boundary_metrics(pred, true, patch_size)
                rows.append({**metrics, **roi, **boundary})
            # patch metrics subject mean
            patch_recon = recon_patches.reshape(-1, 1, *patch_size)
            patch_true = patches.reshape(-1, 1, *patch_size)
            pm = compute_image_metrics_detailed(patch_recon, patch_true, mask=None, data_range=2.0)
            patch_rows.append({k: float(pm[k]) for k in ["mae", "rmse", "psnr", "ssim"]})
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    patch_summary = {k: float(np.mean([r[k] for r in patch_rows])) for k in patch_rows[0]}
    latent_cat = torch.cat(latents, dim=0)
    latent_sum = latent_statistics(latent_cat)
    latent_sum["inactive_channel_ratio"] = float((latent_cat.flatten(2).std(dim=2, unbiased=False) < 1e-3).float().mean().item())
    if first_visual is not None:
        gt, pred = first_visual
        center = [s // 2 for s in gt.shape[-3:]]
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        rows_idx = [center[0], center[1], center[2]]
        for r, axis in enumerate([2, 3, 4]):
            gt_slice = gt[0, 0].select(axis - 2, rows_idx[r]).numpy()
            pr_slice = pred[0, 0].select(axis - 2, rows_idx[r]).numpy()
            err = abs(pr_slice - gt_slice)
            axes[r, 0].imshow(gt_slice, cmap='magma')
            axes[r, 1].imshow(pr_slice, cmap='magma')
            axes[r, 2].imshow(err, cmap='magma')
            axes[r, 0].set_ylabel(['axial','coronal','sagittal'][r])
            for c in range(3): axes[r, c].axis('off')
        axes[0, 0].set_title('GT'); axes[0, 1].set_title('Recon'); axes[0, 2].set_title('AbsErr')
        fig.tight_layout()
        fig.savefig(out_dir / 'visualizations' / f'__vis_{method}.png', dpi=150)
        plt.close(fig)
    return summary, patch_summary, latent_sum, rows, patch_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device(config["training"].get("device", "cuda"))
    out_dir = REPO_ROOT / config["experiment"]["output_dir"]
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    image_size = tuple(int(v) for v in config["data"]["image_size"])
    patch_size = tuple(int(v) for v in config["data"]["patch_size"])
    patch_coords = build_patch_grid(image_size, patch_size)
    train_loader = build_loader(config, "train")
    val_loader = build_val_loader(config)
    method = str(config["experiment"]["method"]).lower()
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))

    if args.smoke_test:
        batch = next(iter(train_loader))
        pet = batch[str(config["data"].get("reconstruction_key", "pet"))].to(device).float()
        patches = extract_patches(pet, patch_coords, patch_size)
        patch_chunk = patches[:, : int(config["training"]["patch_batch_size"])].reshape(-1, 1, *patch_size)
        recon, payload = recon_from_model(model, method, patch_chunk)
        out = {"recon_shape": list(recon.shape)}
        if method == "klvae":
            out.update({"mu_shape": list(payload["mu"].shape), "logvar_shape": list(payload["logvar"].shape), "z_shape": list(payload["z"].shape), "kl": float(kl_loss(payload["mu"], payload["logvar"]).item())})
        elif method == "vqvae":
            out.update({"z_e_shape": list(payload["z_e"].shape), "z_q_shape": list(payload["z_q"].shape), "indices_shape": list(payload["indices"].shape), "codebook_loss": float(payload["codebook"].item()), "commitment_loss": float(payload["commit"].item())})
        else:
            out.update({"z_shape": list(payload["z"].shape)})
        print(json.dumps(out, indent=2, ensure_ascii=True))
        return

    epochs = int(config["training"]["epochs"])
    patch_batch_size = int(config["training"]["patch_batch_size"])
    reconstruction_key = str(config["data"].get("reconstruction_key", "pet"))
    best = float('-inf')
    for epoch in range(1, epochs + 1):
        model.train()
        train_tot = train_rec = train_kl = train_reg = 0.0
        n = 0
        beta = float(config.get("loss", {}).get("kl", {}).get("max_weight", 0.0))
        if method == "klvae":
            klcfg = config["loss"]["kl"]
            if bool(klcfg.get("enabled", True)):
                if epoch < int(klcfg["warmup_start_epoch"]):
                    beta = 0.0
                else:
                    warm = max(1, int(klcfg["warmup_epochs"]))
                    beta = float(klcfg["max_weight"]) * min(1.0, max(0.0, (epoch - int(klcfg["warmup_start_epoch"]) + 1) / warm))
        for batch in train_loader:
            pet = batch[reconstruction_key].to(device).float()
            patches = extract_patches(pet, patch_coords, patch_size)
            optimizer.zero_grad(set_to_none=True)
            recon_chunks = []
            rec_terms = []
            reg_terms = []
            for start in range(0, len(patch_coords), patch_batch_size):
                end = min(start + patch_batch_size, len(patch_coords))
                patch_chunk = patches[:, start:end].reshape(-1, 1, *patch_size)
                recon, payload = recon_from_model(model, method, patch_chunk)
                recon_chunks.append(recon)
                rec_terms.append(torch.abs(recon - patch_chunk).mean())
                if method == "klvae":
                    reg_terms.append(kl_loss(payload["mu"], payload["logvar"]))
                elif method == "vqvae":
                    reg_terms.append(payload["codebook"] + float(config["model"].get("commitment_weight", 0.25)) * payload["commit"])
                elif method == "wae":
                    prior = torch.randn_like(payload["z"])
                    reg_terms.append(mmd_imq(payload["z"], prior))
            recon_patches = torch.cat(recon_chunks, dim=0).view(pet.shape[0], len(patch_coords), 1, *patch_size)
            recon_full = stitch_patches(recon_patches, patch_coords, image_size)
            loss_patch = torch.stack(rec_terms).mean()
            loss_full = torch.abs(recon_full - pet).mean()
            loss_rec = float(config["loss"]["lambda_patch_recon"]) * loss_patch + float(config["loss"]["lambda_full_recon"]) * loss_full
            loss_reg = torch.stack(reg_terms).mean() if reg_terms else loss_patch.new_zeros(())
            if method == "klvae":
                loss_total = loss_rec + beta * loss_reg
            elif method == "vqvae":
                loss_total = loss_rec + float(config["model"].get("vq_weight", 1.0)) * loss_reg
            elif method == "wae":
                loss_total = loss_rec + float(config["model"].get("mmd_weight", 1.0)) * loss_reg
            else:
                loss_total = loss_rec
            loss_total.backward()
            optimizer.step()
            b = int(pet.shape[0])
            n += b
            train_tot += float(loss_total.item()) * b
            train_rec += float(loss_rec.item()) * b
            train_kl += float(loss_reg.item()) * b
        model.eval()
        summary, patch_summary, latent_summary, rows, patch_rows = eval_model(model, val_loader, device, config, method, patch_coords, patch_size, image_size, patch_batch_size, out_dir)
        payload = {"epoch": epoch, "summary": summary, "patch": patch_summary, "latent": latent_summary}
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, method, epoch, config, payload)
        if summary["ssim"] > best:
            best = summary["ssim"]
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, method, epoch, config, payload)


if __name__ == "__main__":
    main()
