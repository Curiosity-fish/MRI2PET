"""Exp037E: Exp037C initialized from 037C plus frozen BSFE ZeroConv adapters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.exp026.utils.patch_codec import build_patch_grid, extract_patches, stitch_patches
from exp032e.utils import ae_decode, ae_encode, load_autoencoder
from metrics.image_metrics import compute_image_metrics_detailed
from exp.exp037E.exp037e_global_bsfe_unet3d import Exp037EGlobalBSFEUNet3D
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from models.exp037c_global_losses import compute_global_losses

def decode_pred(ae_model, z_pred, pet_shape, latent_patch_coords, latent_patch_size, latent_full_size):
    z_patches = extract_patches(z_pred, latent_patch_coords, latent_patch_size)
    z_full = stitch_patches(z_patches, latent_patch_coords, latent_full_size)
    return ae_decode(ae_model, z_full)
def infer_feature_type_ids(table_cols: list[str]) -> tuple[list[int], dict[int, str]]:
    label_map = {0: "demographic", 1: "genetic", 2: "cognitive", 3: "functional_psych", 4: "other"}
    type_ids = []
    for name in table_cols:
        lower = str(name).lower()
        if any(key in lower for key in ["age", "gender", "sex", "educ", "race", "eth", "marry", "hand"]):
            type_ids.append(0)
        elif "apoe" in lower:
            type_ids.append(1)
        elif any(key in lower for key in ["mmse", "adas", "cdr", "ravlt", "moca", "digit", "trail", "clock", "boston", "animal", "lm"]):
            type_ids.append(2)
        elif any(key in lower for key in ["faq", "ecog", "gds", "npi"]):
            type_ids.append(3)
        else:
            type_ids.append(4)
    return type_ids, label_map
from train.train_exp032j_sampling_aligned import build_dataset


HIERARCHY_NAMES = ("enc1", "enc2", "mid", "dec2", "dec1")


def parse_args():
    p = argparse.ArgumentParser(description="Train Exp037B_v3_T6")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def load_config(path: Path):
    with Path(path).open(encoding="utf-8") as f:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv(path: Path, row) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def resolve(root: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def build_loaders(cfg, distributed=False):
    train_ds = build_dataset(cfg, "train")

    nw = int(cfg["training"].get("num_workers", 0))
    batch_size = int(cfg["training"]["batch_size"])

    common = dict(
        num_workers=nw,
        pin_memory=bool(cfg["training"].get("pin_memory", False)),
        persistent_workers=nw > 0,
    )

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )

    val_sets = [
        build_dataset(cfg, split, train_dataset=train_ds)
        for split in cfg["evaluation"]["val_splits"]
    ]
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )

    small_count = min(
        len(val_ds),
        int(cfg["evaluation"].get("small_val_subjects", 20)),
    )
    small_loader = DataLoader(
        Subset(val_ds, range(small_count)),
        batch_size=batch_size,
        shuffle=False,
        **common,
    )

    return train_ds, train_loader, val_loader, small_loader, train_sampler


def mri_endpoint(mri: torch.Tensor, latent_channels: int) -> torch.Tensor:
    x = F.avg_pool3d(mri, kernel_size=4, stride=4)
    return x.repeat(1, latent_channels, 1, 1, 1)


def decode(ae, z, image_shape, latent_coords, latent_patch_size, latent_full_size):
    return decode_pred(
        ae,
        z,
        image_shape,
        latent_coords,
        latent_patch_size,
        latent_full_size,
    )


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state:
        return state
    if all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_baseline_unet_checkpoint(
    model: Exp037EGlobalBSFEUNet3D,
    checkpoint_path: Path,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    state = strip_module_prefix(state)

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current
        and current[key].shape == value.shape
        and not key.startswith(model.TABLE_PREFIXES)
    }
    current.update(compatible)
    model.load_state_dict(current, strict=True)

    return {
        "checkpoint": str(checkpoint_path),
        "loaded_backbone_tensors": len(compatible),
        "total_model_tensors": len(current),
    }


def stage_for_epoch(epoch: int, stage_a_end: int, stage_b_end: int) -> str:
    if epoch <= stage_a_end:
        return "table_warmup"
    if epoch <= stage_b_end:
        return "semantic_finetune"
    return "full_finetune"


def per_sample_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().flatten(1).mean(dim=1)



def feature_balanced_masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T6 HCSPL over observed Table features.

    Why feature-balanced?
    ---------------------
    Some clinical variables are observed more often than others. A plain
    element-wise average would let frequently observed variables dominate.
    Here each feature contributes its own observed-sample mean, then valid
    features are averaged equally.

    Returns:
        huber_loss: scalar HCSPL
        mae: scalar feature-balanced MAE diagnostic
        observed_fraction: scalar fraction of observed entries
    """
    if pred.shape != target.shape or pred.shape != missing_mask.shape:
        raise ValueError(
            "pred/target/missing_mask must have identical [B,F] shape, got "
            f"{tuple(pred.shape)}, {tuple(target.shape)}, "
            f"{tuple(missing_mask.shape)}"
        )

    beta = float(beta)
    if beta <= 0:
        raise ValueError("HCSPL beta must be > 0")

    observed = (missing_mask <= 0.5).to(pred.dtype)
    clean_target = torch.where(
        observed > 0,
        target,
        torch.zeros_like(target),
    )

    abs_error = (pred - clean_target).abs()

    # Explicit Huber implementation keeps compatibility across PyTorch builds.
    quadratic = torch.minimum(
        abs_error,
        abs_error.new_tensor(beta),
    )
    linear = abs_error - quadratic
    element_huber = 0.5 * quadratic.square() / beta + linear

    feature_count = observed.sum(dim=0)
    valid_feature = feature_count > 0

    if not bool(valid_feature.any()):
        zero = pred.sum() * 0.0
        return zero, zero.detach(), observed.mean().detach()

    feature_huber = (
        (element_huber * observed).sum(dim=0)
        / feature_count.clamp_min(1.0)
    )
    feature_mae = (
        (abs_error * observed).sum(dim=0)
        / feature_count.clamp_min(1.0)
    )

    huber_loss = feature_huber[valid_feature].mean()
    mae = feature_mae[valid_feature].mean()
    observed_fraction = observed.mean()

    return huber_loss, mae, observed_fraction


def make_shuffle_permutation(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size <= 1:
        return torch.arange(batch_size, device=device)
    return torch.roll(torch.arange(batch_size, device=device), shifts=1)


def train_step(
    model,
    ae,
    batch,
    cfg,
    device,
    latent_coords,
    latent_patch_size,
    latent_full_size,
    *,
    bsfe,
    use_condition_ranking: bool,
    use_clinical_semantic_loss: bool,
):
    pet = batch["pet"].to(device, non_blocking=True).float()
    mri = batch["mri"].to(device, non_blocking=True).float()
    table = batch["table"].to(device, non_blocking=True).float()
    table_mask = batch["table_mask"].to(device, non_blocking=True).float()

    x1 = mri_endpoint(mri, int(cfg["model"]["latent_channels"]))
    t = torch.ones(pet.shape[0], device=device, dtype=torch.long)

    with torch.no_grad():
        bsfe_features = bsfe(mri)
        bsfe_pyramid = bsfe_features["pyramid"]
    z_pred, aux = model(
        x1,
        t,
        table,
        table_mask,
        return_aux=True,
        bsfe_pyramid=bsfe_pyramid,
    )

    recon = decode(
        ae,
        z_pred,
        pet.shape,
        latent_coords,
        latent_patch_size,
        latent_full_size,
    )

    correct_err = per_sample_mae(recon, pet)
    image_l1 = correct_err.mean()

    rank_loss = image_l1.new_zeros(())
    delta_gt_mae_shuffle = image_l1.new_zeros(())

    rank_cfg = cfg.get("condition_ranking", {})
    rank_weight = float(rank_cfg.get("weight", 0.0))
    rank_margin = float(rank_cfg.get("margin", 5e-4))

    if use_condition_ranking and rank_weight > 0 and table.shape[0] > 1:
        perm = make_shuffle_permutation(table.shape[0], device)

        z_shuffle = model(
            x1,
            t,
            table[perm],
            table_mask[perm],
            bsfe_pyramid=bsfe_pyramid,
        )
        recon_shuffle = decode(
            ae,
            z_shuffle,
            pet.shape,
            latent_coords,
            latent_patch_size,
            latent_full_size,
        )

        shuffle_err = per_sample_mae(recon_shuffle, pet)

        rank_loss = F.relu(
            rank_margin + correct_err - shuffle_err
        ).mean()
        delta_gt_mae_shuffle = (shuffle_err - correct_err).mean()

    # ------------------------------------------------------------
    # T6 HCSPL: preserve clinical semantics in FINAL refined tokens T5.
    # ------------------------------------------------------------
    semantic_cfg = cfg.get("clinical_semantic_loss", {})
    semantic_weight = float(semantic_cfg.get("weight", 0.05))
    semantic_beta = float(semantic_cfg.get("beta", 1.0))

    clinical_semantic_loss = image_l1.new_zeros(())
    clinical_semantic_mae = image_l1.new_zeros(())
    clinical_observed_fraction = image_l1.new_zeros(())

    if use_clinical_semantic_loss and semantic_weight > 0:
        clinical_semantic_loss, clinical_semantic_mae, clinical_observed_fraction = (
            feature_balanced_masked_huber(
                aux["clinical_pred"],
                table,
                table_mask,
                beta=semantic_beta,
            )
        )

    bsfe_weight = float(cfg.get("loss", {}).get("bsfe_latent_weight", 0.0))
    bsfe_target = bsfe_features.get("latent_prediction")
    bsfe_loss = image_l1.new_zeros(())
    if bsfe_target is not None and bsfe_weight > 0:
        if bsfe_target.shape != z_pred.shape:
            bsfe_target = F.interpolate(bsfe_target, size=z_pred.shape[-3:], mode="trilinear", align_corners=False)
        if bsfe_target.shape[1] != z_pred.shape[1]:
            bsfe_target = bsfe_target.repeat(1, z_pred.shape[1], 1, 1, 1)
        bsfe_loss = F.smooth_l1_loss(z_pred, bsfe_target)

    global_weighted = image_l1.new_zeros(())
    global_losses = {key: image_l1.new_zeros(()) for key in ("gca", "gms", "gms_g3", "gms_g2", "gr")}
    global_cfg = cfg.get("global", {})
    if bool(global_cfg.get("enabled", False)):
        with torch.no_grad():
            z_target, _ = ae_encode(ae, pet, cfg["ae"]["kind"])
        global_result = compute_global_losses(
            aux, z_target,
            gca_weight=float(global_cfg.get("context_weight", 0.05)),
            gms_weight=float(global_cfg.get("multiscale_weight", 0.10)),
            gr_weight=float(global_cfg.get("relation_weight", 0.02)),
            g3_scale_weight=float(global_cfg.get("g3_scale_weight", 0.7)),
            g2_scale_weight=float(global_cfg.get("g2_scale_weight", 0.3)),
            smooth_l1_beta=float(global_cfg.get("smooth_l1_beta", 1.0)),
        )
        global_weighted = global_result["global_weighted"]
        global_losses = global_result

    total_loss = (
        image_l1
        + rank_weight * rank_loss
        + semantic_weight * clinical_semantic_loss
        + bsfe_weight * bsfe_loss
        + global_weighted
    )

    train_aux = {
        **aux,
        "image_l1": image_l1.detach(),
        "condition_rank_loss": rank_loss.detach(),
        "delta_gt_mae_shuffle_train": delta_gt_mae_shuffle.detach(),
        "clinical_semantic_loss": clinical_semantic_loss.detach(),
        "clinical_semantic_mae": clinical_semantic_mae.detach(),
        "clinical_observed_fraction": clinical_observed_fraction.detach(),
        "total_loss": total_loss.detach(),
        "bsfe_loss": bsfe_loss.detach(),
        "global_weighted": global_weighted.detach(),
        **{f"global_{key}": value.detach() for key, value in global_losses.items()},
    }

    return total_loss, recon, pet, train_aux


@torch.no_grad()
def evaluate(
    model,
    ae,
    loader,
    cfg,
    device,
    latent_coords,
    latent_patch_size,
    latent_full_size,
    *,
    bsfe,
    max_batches=None,
):
    model.eval()
    rows = []

    for batch_idx, batch in enumerate(loader):
        pet = batch["pet"].to(device, non_blocking=True).float()
        mri = batch["mri"].to(device, non_blocking=True).float()
        table = batch["table"].to(device, non_blocking=True).float()
        table_mask = batch["table_mask"].to(device, non_blocking=True).float()

        x1 = mri_endpoint(mri, int(cfg["model"]["latent_channels"]))
        t = torch.ones(pet.shape[0], device=device, dtype=torch.long)
        bsfe_pyramid = bsfe(mri)["pyramid"]

        z_pred, aux = model(
            x1,
            t,
            table,
            table_mask,
            return_aux=True,
            bsfe_pyramid=bsfe_pyramid,
        )
        recon = decode(
            ae,
            z_pred,
            pet.shape,
            latent_coords,
            latent_patch_size,
            latent_full_size,
        )

        semantic_cfg = cfg.get("clinical_semantic_loss", {})
        val_sem_loss, val_sem_mae, val_observed_fraction = (
            feature_balanced_masked_huber(
                aux["clinical_pred"],
                table,
                table_mask,
                beta=float(semantic_cfg.get("beta", 1.0)),
            )
        )

        mask = batch["brain_mask"].to(device, non_blocking=True).float()
        metrics = compute_image_metrics_detailed(
            recon,
            pet,
            mask=mask,
            data_range=2.0,
        )

        row = {
            "image_l1": float(F.l1_loss(recon, pet)),
            "ssim": float(metrics["ssim"]),
            "psnr": float(metrics["psnr"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "brain_ssim": float(metrics["brain_ssim"]),
            "brain_psnr": float(metrics["brain_psnr"]),
            "brain_mae": float(metrics["brain_mae"]),
            "table_injection_ratio": float(aux["table_injection_ratio"]),
            "hierarchical_injection_ratio_max": float(
                aux["hierarchical_injection_ratio_max"]
            ),
            "terminal_dec0_injection_ratio": float(
                aux["terminal_dec0_injection_ratio"]
            ),
            "subject_token_norm": float(aux["subject_token_norm"]),
            "final_table_token_norm": float(aux["final_table_token_norm"]),
            "table_reliability": float(aux["table_reliability"]),

            # T6 semantic-preservation validation metrics.
            "clinical_semantic_loss": float(val_sem_loss),
            "clinical_semantic_mae": float(val_sem_mae),
            "clinical_observed_fraction": float(val_observed_fraction),
            "clinical_pred_abs_mean": float(aux["clinical_pred_abs_mean"]),
        }

        for name in HIERARCHY_NAMES:
            row[f"{name}_injection_ratio"] = float(
                aux[f"{name}_injection_ratio"]
            )
            row[f"{name}_table_update_ratio"] = float(
                aux[f"{name}_table_update_ratio"]
            )

        rows.append(row)

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    if not rows:
        raise RuntimeError("Validation loader produced no batches.")

    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


@torch.no_grad()
def evaluate_table_sensitivity(
    model,
    ae,
    loader,
    cfg,
    device,
    latent_coords,
    latent_patch_size,
    latent_full_size,
    max_batches: int = 5,
):
    model.eval()

    keys = [
        "latent_l1_shuffle",
        "latent_l1_missing",
        "latent_l1_disabled",
        "pet_mae_shuffle",
        "pet_mae_missing",
        "pet_mae_disabled",
        "pet_mae_missing_vs_disabled",
        "gt_mae_correct",
        "gt_mae_shuffle",
        "gt_mae_missing",
        "gt_mae_disabled",
        "delta_gt_mae_shuffle",
        "delta_gt_mae_missing",
        "delta_gt_mae_disabled",
        "correct_better_shuffle_rate",
        "gt_ssim_correct",
        "gt_ssim_shuffle",
        "delta_gt_ssim_correct_minus_shuffle",
        "table_injection_ratio",
        "terminal_dec0_injection_ratio",
        "subject_token_norm",
        "final_table_token_norm",
    ]
    for name in HIERARCHY_NAMES:
        keys += [
            f"{name}_injection_ratio",
            f"{name}_table_update_ratio",
        ]

    collected = {k: [] for k in keys}

    for batch_idx, batch in enumerate(loader):
        pet = batch["pet"].to(device, non_blocking=True).float()
        mri = batch["mri"].to(device, non_blocking=True).float()
        table = batch["table"].to(device, non_blocking=True).float()
        table_mask = batch["table_mask"].to(device, non_blocking=True).float()
        brain_mask = batch["brain_mask"].to(device, non_blocking=True).float()

        x1 = mri_endpoint(mri, int(cfg["model"]["latent_channels"]))
        t = torch.ones(pet.shape[0], device=device, dtype=torch.long)

        z_orig, aux = model(
            x1, t, table, table_mask, return_aux=True
        )

        zero_table = torch.zeros_like(table)
        missing_mask = torch.ones_like(table_mask)

        z_missing = model(
            x1, t, zero_table, missing_mask
        )
        z_disabled = model(
            x1, t, table, table_mask, disable_table=True
        )

        z_shuffle = None
        if table.shape[0] > 1:
            perm = make_shuffle_permutation(table.shape[0], device)
            z_shuffle = model(
                x1,
                t,
                table[perm],
                table_mask[perm],
            )

        pet_orig = decode(
            ae, z_orig, pet.shape, latent_coords, latent_patch_size, latent_full_size
        )
        pet_missing = decode(
            ae, z_missing, pet.shape, latent_coords, latent_patch_size, latent_full_size
        )
        pet_disabled = decode(
            ae, z_disabled, pet.shape, latent_coords, latent_patch_size, latent_full_size
        )
        pet_shuffle = (
            decode(
                ae,
                z_shuffle,
                pet.shape,
                latent_coords,
                latent_patch_size,
                latent_full_size,
            )
            if z_shuffle is not None
            else None
        )

        collected["latent_l1_missing"].append(
            float(F.l1_loss(z_orig, z_missing))
        )
        collected["latent_l1_disabled"].append(
            float(F.l1_loss(z_orig, z_disabled))
        )
        collected["pet_mae_missing"].append(
            float(F.l1_loss(pet_orig, pet_missing))
        )
        collected["pet_mae_disabled"].append(
            float(F.l1_loss(pet_orig, pet_disabled))
        )
        collected["pet_mae_missing_vs_disabled"].append(
            float(F.l1_loss(pet_missing, pet_disabled))
        )

        if z_shuffle is not None:
            collected["latent_l1_shuffle"].append(
                float(F.l1_loss(z_orig, z_shuffle))
            )
            collected["pet_mae_shuffle"].append(
                float(F.l1_loss(pet_orig, pet_shuffle))
            )

        err_correct = per_sample_mae(pet_orig, pet)
        err_missing = per_sample_mae(pet_missing, pet)
        err_disabled = per_sample_mae(pet_disabled, pet)

        collected["gt_mae_correct"].extend(err_correct.cpu().tolist())
        collected["gt_mae_missing"].extend(err_missing.cpu().tolist())
        collected["gt_mae_disabled"].extend(err_disabled.cpu().tolist())
        collected["delta_gt_mae_missing"].extend(
            (err_missing - err_correct).cpu().tolist()
        )
        collected["delta_gt_mae_disabled"].extend(
            (err_disabled - err_correct).cpu().tolist()
        )

        metrics_correct = compute_image_metrics_detailed(
            pet_orig,
            pet,
            mask=brain_mask,
            data_range=2.0,
        )
        collected["gt_ssim_correct"].append(
            float(metrics_correct["ssim"])
        )

        if pet_shuffle is not None:
            err_shuffle = per_sample_mae(pet_shuffle, pet)
            delta = err_shuffle - err_correct

            collected["gt_mae_shuffle"].extend(err_shuffle.cpu().tolist())
            collected["delta_gt_mae_shuffle"].extend(delta.cpu().tolist())
            collected["correct_better_shuffle_rate"].extend(
                (delta > 0).float().cpu().tolist()
            )

            metrics_shuffle = compute_image_metrics_detailed(
                pet_shuffle,
                pet,
                mask=brain_mask,
                data_range=2.0,
            )
            ssim_shuffle = float(metrics_shuffle["ssim"])
            collected["gt_ssim_shuffle"].append(ssim_shuffle)
            collected["delta_gt_ssim_correct_minus_shuffle"].append(
                float(metrics_correct["ssim"]) - ssim_shuffle
            )

        collected["table_injection_ratio"].append(
            float(aux["table_injection_ratio"])
        )
        collected["terminal_dec0_injection_ratio"].append(
            float(aux["terminal_dec0_injection_ratio"])
        )
        collected["subject_token_norm"].append(
            float(aux["subject_token_norm"])
        )
        collected["final_table_token_norm"].append(
            float(aux["final_table_token_norm"])
        )
        for name in HIERARCHY_NAMES:
            collected[f"{name}_injection_ratio"].append(
                float(aux[f"{name}_injection_ratio"])
            )
            collected[f"{name}_table_update_ratio"].append(
                float(aux[f"{name}_table_update_ratio"])
            )

        if batch_idx + 1 >= max_batches:
            break

    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in collected.items()
    }


def _debug_scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 1:
            return float(value.float().cpu())
        return {
            "shape": list(value.shape),
            "finite_fraction": float(
                torch.isfinite(value).float().mean().cpu()
            ),
        }
    return value


def assert_finite_forward(
    loss: torch.Tensor,
    aux: dict,
    *,
    epoch: int,
    batch_idx: int,
    global_step: int,
) -> None:
    if not torch.isfinite(loss).all():
        snapshot = {
            key: _debug_scalar(value)
            for key, value in aux.items()
            if torch.is_tensor(value) and value.numel() == 1
        }
        raise FloatingPointError(
            "Non-finite loss BEFORE backward. "
            f"epoch={epoch}, batch_idx={batch_idx}, global_step={global_step}, "
            f"diagnostics={snapshot}"
        )

    for key, value in aux.items():
        if not torch.is_tensor(value):
            continue
        if value.dtype == torch.bool:
            continue
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite forward diagnostic '{key}'. "
                f"epoch={epoch}, batch_idx={batch_idx}, global_step={global_step}, "
                f"value={_debug_scalar(value)}"
            )

    if not bool(aux["bica_all_finite"].detach().cpu()):
        raise FloatingPointError(
            "T5 H-BiCA internal finite check failed. "
            f"epoch={epoch}, batch_idx={batch_idx}, global_step={global_step}"
        )


def assert_finite_gradients(
    raw_model,
    *,
    epoch: int,
    batch_idx: int,
    global_step: int,
) -> None:
    for name, param in raw_model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            finite_fraction = float(
                torch.isfinite(param.grad.detach()).float().mean().cpu()
            )
            raise FloatingPointError(
                "Non-finite gradient AFTER backward. "
                f"epoch={epoch}, batch_idx={batch_idx}, global_step={global_step}, "
                f"parameter={name}, finite_fraction={finite_fraction:.6f}"
            )


def assert_finite_trainable_parameters(
    raw_model,
    *,
    epoch: int,
    batch_idx: int,
    global_step: int,
) -> None:
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if not torch.isfinite(param).all():
            finite_fraction = float(
                torch.isfinite(param.detach()).float().mean().cpu()
            )
            raise FloatingPointError(
                "Non-finite parameter AFTER optimizer.step(). "
                f"epoch={epoch}, batch_idx={batch_idx}, global_step={global_step}, "
                f"parameter={name}, finite_fraction={finite_fraction:.6f}"
            )


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_ssim: float,
    cfg,
    metrics,
    stage: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_ssim": best_ssim,
            "config": cfg,
            "metrics": metrics,
            "training_stage": stage,
        },
        path,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(cfg["experiment"].get("seed", 42)))

    out = REPO_ROOT / cfg["experiment"]["output_dir"]
    if rank == 0:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(parents=True, exist_ok=True)

    image_size = tuple(cfg["data"]["image_size"])
    patch_size = tuple(cfg["data"]["patch_size"])
    latent_full = tuple(v // 4 for v in image_size)
    latent_patch = tuple(v // 4 for v in patch_size)
    latent_coords = build_patch_grid(latent_full, latent_patch)

    train_ds, train_loader, val_loader, small_loader, train_sampler = (
        build_loaders(cfg, distributed=distributed)
    )

    feature_type_ids, feature_type_map = infer_feature_type_ids(
        train_ds.table_cols
    )

    ae = load_autoencoder(
        cfg["ae"]["kind"],
        resolve(REPO_ROOT, cfg["ae"]["checkpoint_path"]),
        device,
    )
    ae.eval()
    for param in ae.parameters():
        param.requires_grad_(False)

    model_cfg = cfg["model"]
    table_cfg = cfg["table"]

    model = Exp037EGlobalBSFEUNet3D(
        table_num_features=len(train_ds.table_cols),
        feature_type_ids=feature_type_ids,
        in_channels=int(model_cfg["latent_channels"]),
        out_channels=int(model_cfg["latent_channels"]),
        channels=tuple(model_cfg["channels"]),
        time_dim=int(model_cfg["time_dim"]),
        norm_groups=int(model_cfg["norm_groups"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        table_token_dim=int(table_cfg.get("token_dim", 128)),
        table_depth=int(table_cfg.get("depth", 2)),
        table_heads=int(table_cfg.get("heads", 4)),
        table_ff_mult=int(table_cfg.get("ff_mult", 4)),
        table_dropout=float(table_cfg.get("dropout", 0.0)),
        table_num_feature_types=int(table_cfg.get("num_feature_types", 5)),
        bica_heads=int(table_cfg.get("bica_heads", 4)),
        bica_out_init_std=float(table_cfg.get("bica_out_init_std", 0.005)),
        enc1_max_injection_ratio=float(
            table_cfg.get("enc1_max_injection_ratio", 0.01)
        ),
        enc1_table_update_scale=float(
            table_cfg.get("enc1_table_update_scale", 0.05)
        ),
        enc2_max_injection_ratio=float(
            table_cfg.get("enc2_max_injection_ratio", 0.02)
        ),
        enc2_table_update_scale=float(
            table_cfg.get("enc2_table_update_scale", 0.075)
        ),
        mid_max_injection_ratio=float(
            table_cfg.get("mid_max_injection_ratio", 0.04)
        ),
        mid_table_update_scale=float(
            table_cfg.get("mid_table_update_scale", 0.10)
        ),
        dec2_max_injection_ratio=float(
            table_cfg.get("dec2_max_injection_ratio", 0.02)
        ),
        dec2_table_update_scale=float(
            table_cfg.get("dec2_table_update_scale", 0.075)
        ),
        dec1_max_injection_ratio=float(
            table_cfg.get("dec1_max_injection_ratio", 0.01)
        ),
        dec1_table_update_scale=float(
            table_cfg.get("dec1_table_update_scale", 0.05)
        ),
        terminal_dec0_max_injection_ratio=float(
            table_cfg.get("terminal_dec0_max_injection_ratio", 0.005)
        ),
        table_max_update_ratio=float(
            table_cfg.get("table_max_update_ratio", 0.10)
        ),
        image_residual_scale=float(
            table_cfg.get("image_residual_scale", 0.10)
        ),
        clinical_decoder_hidden_dim=int(
            table_cfg.get("clinical_decoder_hidden_dim", 64)
        ),
        bsfe_channels=tuple(model_cfg.get("bsfe_channels", [48, 96, 192, 384])),
        global_vit_enabled=bool(cfg.get("global", {}).get("enabled", False)),
        global_token_dim=int(cfg.get("global", {}).get("token_dim", 256)),
        global_vit_depth=int(cfg.get("global", {}).get("depth", 3)),
        global_vit_heads=int(cfg.get("global", {}).get("heads", 8)),
        global_vit_mlp_ratio=float(cfg.get("global", {}).get("mlp_ratio", 4.0)),
        global_vit_dropout=float(cfg.get("global", {}).get("dropout", 0.0)),
        global_vit_attention_dropout=float(cfg.get("global", {}).get("attention_dropout", 0.0)),
        global_fusion_gate_bias=float(cfg.get("global", {}).get("fusion_gate_bias", -2.0)),
        global_fusion_init_alpha=float(cfg.get("global", {}).get("fusion_init_alpha", 0.1)),
    ).to(device)

    bsfe, _bsfe_head = build_bsfe_models(cfg, latent_full, device)
    bsfe_ckpt = torch.load(resolve(REPO_ROOT, cfg["bsfe"]["checkpoint_path"]), map_location=device, weights_only=False)
    bsfe.load_state_dict(bsfe_ckpt["bsfe"], strict=True)
    for p in bsfe.parameters():
        p.requires_grad_(False)
    bsfe.eval()

    init_report = {"checkpoint": None, "loaded_backbone_tensors": 0}
    init_cfg = cfg.get("initialization", {})
    if init_cfg.get("unet_checkpoint"):
        init_report = load_baseline_unet_checkpoint(
            model,
            resolve(REPO_ROOT, init_cfg["unet_checkpoint"]),
        )

    raw_model = model
    groups = raw_model.split_parameter_groups()

    table_lr = float(cfg["training"].get("table_lr", 1e-4))
    semantic_lr = float(cfg["training"].get("semantic_lr", 2e-5))
    shallow_lr = float(cfg["training"].get("shallow_lr", 5e-6))

    optimizer = torch.optim.AdamW(
        [
            {"params": groups["table"], "lr": table_lr, "name": "table"},
            {"params": groups["semantic"], "lr": semantic_lr, "name": "semantic"},
            {"params": groups["shallow"], "lr": shallow_lr, "name": "shallow"},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=tuple(cfg["training"].get("betas", [0.9, 0.999])),
    )

    epochs = 1 if args.smoke_test else int(cfg["training"]["epochs"])
    stage_a_end = int(cfg["training"].get("stage_a_end_epoch", 3))
    stage_b_end = int(cfg["training"].get("stage_b_end_epoch", 10))

    full_stage_start = stage_b_end + 1
    full_stage_epochs = max(epochs - full_stage_start + 1, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=full_stage_epochs,
    )

    amp_mode = cfg["training"].get("amp", "bf16")
    amp_enabled = amp_mode != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_enabled and amp_mode == "fp16"),
    )

    numerics_cfg = cfg.get("numerics", {})
    fail_fast_nonfinite = bool(
        numerics_cfg.get("fail_fast_nonfinite", True)
    )
    gradient_check_every = max(
        int(numerics_cfg.get("gradient_check_every_n_steps", 1)),
        1,
    )
    parameter_check_every = max(
        int(numerics_cfg.get("parameter_check_every_n_steps", 10)),
        1,
    )

    start_epoch = 1
    best_ssim = -float("inf")

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_ssim = float(state.get("best_ssim", best_ssim))

    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    report = {
        "experiment": cfg["experiment"]["name"],
        "parameters_total": sum(p.numel() for p in raw_model.parameters()),
        "parameters_table": sum(p.numel() for p in groups["table"]),
        "parameters_semantic": sum(p.numel() for p in groups["semantic"]),
        "parameters_shallow": sum(p.numel() for p in groups["shallow"]),
        "table_features": len(train_ds.table_cols),
        "table_columns": train_ds.table_cols,
        "feature_type_map": feature_type_map,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "initialization": init_report,
        "hierarchy": [
            "Enc1 <-> T0 -> T1",
            "Enc2 <-> T1 -> T2",
            "Mid <-> T2 -> T3",
            "Dec2 <-> T3 -> T4",
            "Dec1 <-> T4 -> T5",
            "Dec0 <- T5 terminal feedback",
        ],
        "loss": "image_l1 + correct_vs_shuffle_ranking + HCSPL",
        "input_mode": "raw_mri_avgpool_repeat",
        "numerical_safety": cfg.get("numerics", {}),
        "clinical_semantic_loss": cfg.get("clinical_semantic_loss", {}),
    }

    if rank == 0:
        print(json.dumps(report, ensure_ascii=False), flush=True)
        (out / "architecture.txt").write_text(
            json.dumps(
                {
                    "model": model_cfg,
                    "table": table_cfg,
                    "training": cfg["training"],
                    "report": report,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    train_csv = out / "logs" / "train_log.csv"
    val_csv = out / "logs" / "val_log.csv"
    sensitivity_csv = out / "logs" / "table_sensitivity.csv"

    stage_metric_names = []
    for name in HIERARCHY_NAMES:
        stage_metric_names += [
            f"{name}_injection_ratio",
            f"{name}_table_update_ratio",
        ]

    if rank == 0:
        ensure_csv(
            train_csv,
            [
                "epoch",
                "stage",
                "total_loss",
                "image_l1",
                "condition_rank_loss",
                "delta_gt_mae_shuffle_train",
                "clinical_semantic_loss",
                "clinical_semantic_mae",
                "clinical_observed_fraction",
                "bsfe_loss",
                "global_weighted",
                "global_gca",
                "global_gms",
                "global_gms_g3",
                "global_gms_g2",
                "global_gr",
                "table_lr",
                "semantic_lr",
                "shallow_lr",
                "table_injection_ratio",
                *stage_metric_names,
                "terminal_dec0_injection_ratio",
                "subject_token_norm",
                "final_table_token_norm",
                "peak_gpu_mem_mb",
                "epoch_time_sec",
            ],
        )

        ensure_csv(
            val_csv,
            [
                "epoch",
                "stage",
                "image_l1",
                "ssim",
                "psnr",
                "mae",
                "rmse",
                "brain_ssim",
                "brain_psnr",
                "brain_mae",
                "table_injection_ratio",
                "hierarchical_injection_ratio_max",
                *stage_metric_names,
                "terminal_dec0_injection_ratio",
                "subject_token_norm",
                "final_table_token_norm",
                "table_reliability",
                "clinical_semantic_loss",
                "clinical_semantic_mae",
                "clinical_observed_fraction",
                "clinical_pred_abs_mean",
            ],
        )

        ensure_csv(
            sensitivity_csv,
            [
                "epoch",
                "stage",
                "latent_l1_shuffle",
                "latent_l1_missing",
                "latent_l1_disabled",
                "pet_mae_shuffle",
                "pet_mae_missing",
                "pet_mae_disabled",
                "pet_mae_missing_vs_disabled",
                "gt_mae_correct",
                "gt_mae_shuffle",
                "gt_mae_missing",
                "gt_mae_disabled",
                "delta_gt_mae_shuffle",
                "delta_gt_mae_missing",
                "delta_gt_mae_disabled",
                "correct_better_shuffle_rate",
                "gt_ssim_correct",
                "gt_ssim_shuffle",
                "delta_gt_ssim_correct_minus_shuffle",
                "table_injection_ratio",
                *stage_metric_names,
                "terminal_dec0_injection_ratio",
                "subject_token_norm",
                "final_table_token_norm",
            ],
        )

    grad_accum = (
        1 if args.smoke_test
        else max(int(cfg["training"].get("grad_accum", 1)), 1)
    )

    rank_cfg = cfg.get("condition_ranking", {})
    ranking_every = max(int(rank_cfg.get("every_n_steps", 1)), 1)
    ranking_start_epoch = int(rank_cfg.get("start_epoch", 1))

    semantic_cfg = cfg.get("clinical_semantic_loss", {})
    semantic_start_epoch = int(semantic_cfg.get("start_epoch", 1))

    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)

        stage = stage_for_epoch(epoch, stage_a_end, stage_b_end)
        raw_model.set_training_stage(stage)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logs = {
            "total_loss": [],
            "image_l1": [],
            "condition_rank_loss": [],
            "delta_gt_mae_shuffle_train": [],
            "clinical_semantic_loss": [],
            "clinical_semantic_mae": [],
            "clinical_observed_fraction": [],
            "table_injection_ratio": [],
            "terminal_dec0_injection_ratio": [],
            "subject_token_norm": [],
            "final_table_token_norm": [],
            "bsfe_loss": [],
            "global_weighted": [],
            "global_gca": [],
            "global_gms": [],
            "global_gms_g3": [],
            "global_gms_g2": [],
            "global_gr": [],
        }
        for metric in stage_metric_names:
            logs[metric] = []

        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, batch in enumerate(train_loader):
            use_rank = (
                epoch >= ranking_start_epoch
                and global_step % ranking_every == 0
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                loss, _, _, aux = train_step(
                    model,
                    ae,
                    batch,
                    cfg,
                    device,
                    latent_coords,
                    latent_patch,
                    latent_full,
                    bsfe=bsfe,
                    use_condition_ranking=use_rank,
                    use_clinical_semantic_loss=(
                        epoch >= semantic_start_epoch
                    ),
                )
                scaled_loss = loss / grad_accum

            if fail_fast_nonfinite:
                assert_finite_forward(
                    loss,
                    aux,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                )

            scaler.scale(scaled_loss).backward()

            if (
                fail_fast_nonfinite
                and global_step % gradient_check_every == 0
            ):
                assert_finite_gradients(
                    raw_model,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                )

            if (
                (batch_idx + 1) % grad_accum == 0
                or batch_idx + 1 == len(train_loader)
                or args.smoke_test
            ):
                scaler.step(optimizer)
                scaler.update()

                if (
                    fail_fast_nonfinite
                    and global_step % parameter_check_every == 0
                ):
                    assert_finite_trainable_parameters(
                        raw_model,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                    )

                optimizer.zero_grad(set_to_none=True)

            for key in logs:
                logs[key].append(float(aux[key]))

            global_step += 1
            if args.smoke_test:
                break

        if epoch >= full_stage_start:
            scheduler.step()

        means = {
            key: float(np.mean(values))
            for key, values in logs.items()
        }
        peak_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        lrs = {
            group["name"]: group["lr"]
            for group in optimizer.param_groups
        }

        if rank == 0:
            append_csv(
                train_csv,
                [
                    epoch,
                    stage,
                    means["total_loss"],
                    means["image_l1"],
                    means["condition_rank_loss"],
                    means["delta_gt_mae_shuffle_train"],
                    means["clinical_semantic_loss"],
                    means["clinical_semantic_mae"],
                    means["clinical_observed_fraction"],
                    means["bsfe_loss"],
                    means["global_weighted"],
                    means["global_gca"],
                    means["global_gms"],
                    means["global_gms_g3"],
                    means["global_gms_g2"],
                    means["global_gr"],
                    lrs["table"],
                    lrs["semantic"],
                    lrs["shallow"],
                    means["table_injection_ratio"],
                    *[means[m] for m in stage_metric_names],
                    means["terminal_dec0_injection_ratio"],
                    means["subject_token_norm"],
                    means["final_table_token_norm"],
                    peak_mb,
                    time.time() - started,
                ],
            )

        if distributed and rank != 0:
            torch.distributed.barrier()
            continue

        eval_limit = 1 if args.smoke_test else None

        small_metrics = evaluate(
            raw_model,
            ae,
            small_loader,
            cfg,
            device,
            latent_coords,
            latent_patch,
            latent_full,
            bsfe=bsfe,
            max_batches=eval_limit,
        )
        full_metrics = evaluate(
            raw_model,
            ae,
            val_loader,
            cfg,
            device,
            latent_coords,
            latent_patch,
            latent_full,
            bsfe=bsfe,
            max_batches=eval_limit,
        )

        sensitivity_batches = (
            1 if args.smoke_test
            else int(cfg["evaluation"].get("table_sensitivity_batches", 5))
        )
        sensitivity = evaluate_table_sensitivity(
            raw_model,
            ae,
            small_loader,
            cfg,
            device,
            latent_coords,
            latent_patch,
            latent_full,
            max_batches=sensitivity_batches,
        )

        append_csv(
            val_csv,
            [
                epoch,
                stage,
                full_metrics["image_l1"],
                full_metrics["ssim"],
                full_metrics["psnr"],
                full_metrics["mae"],
                full_metrics["rmse"],
                full_metrics["brain_ssim"],
                full_metrics["brain_psnr"],
                full_metrics["brain_mae"],
                full_metrics["table_injection_ratio"],
                full_metrics["hierarchical_injection_ratio_max"],
                *[full_metrics[m] for m in stage_metric_names],
                full_metrics["terminal_dec0_injection_ratio"],
                full_metrics["subject_token_norm"],
                full_metrics["final_table_token_norm"],
                full_metrics["table_reliability"],
                full_metrics["clinical_semantic_loss"],
                full_metrics["clinical_semantic_mae"],
                full_metrics["clinical_observed_fraction"],
                full_metrics["clinical_pred_abs_mean"],
            ],
        )

        sensitivity_order = [
            "latent_l1_shuffle",
            "latent_l1_missing",
            "latent_l1_disabled",
            "pet_mae_shuffle",
            "pet_mae_missing",
            "pet_mae_disabled",
            "pet_mae_missing_vs_disabled",
            "gt_mae_correct",
            "gt_mae_shuffle",
            "gt_mae_missing",
            "gt_mae_disabled",
            "delta_gt_mae_shuffle",
            "delta_gt_mae_missing",
            "delta_gt_mae_disabled",
            "correct_better_shuffle_rate",
            "gt_ssim_correct",
            "gt_ssim_shuffle",
            "delta_gt_ssim_correct_minus_shuffle",
            "table_injection_ratio",
            *stage_metric_names,
            "terminal_dec0_injection_ratio",
            "subject_token_norm",
            "final_table_token_norm",
        ]
        append_csv(
            sensitivity_csv,
            [epoch, stage] + [sensitivity[k] for k in sensitivity_order],
        )

        checkpoint_metrics = {
            "small": small_metrics,
            "full": full_metrics,
            "table_sensitivity": sensitivity,
        }

        save_checkpoint(
            out / "checkpoints" / "last.pt",
            epoch,
            raw_model,
            optimizer,
            scheduler,
            best_ssim,
            cfg,
            checkpoint_metrics,
            stage,
        )

        if full_metrics["ssim"] > best_ssim:
            best_ssim = full_metrics["ssim"]
            save_checkpoint(
                out / "checkpoints" / "best_ssim.pt",
                epoch,
                raw_model,
                optimizer,
                scheduler,
                best_ssim,
                cfg,
                checkpoint_metrics,
                stage,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "stage": stage,
                    "train": means,
                    "small": small_metrics,
                    "full": full_metrics,
                    "table_sensitivity": sensitivity,
                    "peak_gpu_mem_mb": peak_mb,
                    "best_ssim": best_ssim,
                    "lr": lrs,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if distributed:
            torch.distributed.barrier()

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
