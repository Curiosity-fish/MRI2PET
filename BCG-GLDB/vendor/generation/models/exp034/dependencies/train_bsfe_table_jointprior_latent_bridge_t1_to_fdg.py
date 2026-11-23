from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridge.residual_bridge import ResidualBridgeDiffusion
from datasets.paired_t1_fdg_table_dataset import PairedT1FDGTableDataset
from engine.trainer_t1_fdg import append_log, prepare_experiment_dir
from metrics.image_metrics import compute_image_metrics_detailed
from models.bsfe_table_jointprior_latent_bridge_denoiser import (
    BSFETableJointPriorLatentBridgeDenoiser3D,
)
from train.train_bsfe_t1_to_fdg import build_models as build_bsfe_models
from train.train_bsfe_t1_to_fdg import infer_latent_shape as infer_bsfe_latent_shape
from train.train_bsfe_t1_to_fdg import load_frozen_dvae as load_bsfe_dvae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BSFE + joint-prior table latent residual bridge.")
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


def build_dataset(
    config: dict,
    split: str,
    tabular_stats: dict | None = None,
    table_cols: list[str] | None = None,
) -> PairedT1FDGTableDataset:
    return PairedT1FDGTableDataset(
        pairs_csv=REPO_ROOT / config["data"]["pairs_csv"],
        split_csv=REPO_ROOT / config["data"]["split_csv"],
        table_csv=config["data"]["table_csv"],
        split=split,
        fold=None,
        spatial_size=tuple(config["data"]["image_size"]),
        tabular_stats=tabular_stats,
        table_cols=table_cols,
        max_samples=config["data"].get(f"{split}_max_samples"),
        standardize_table=bool(config["data"].get("standardize_table", True)),
        add_missing_mask=bool(config["data"].get("add_missing_mask", True)),
        cache_dir=(REPO_ROOT / config["data"]["cache_dir"]) if config["data"].get("cache_dir") else None,
    )


def build_loader(
    dataset: PairedT1FDGTableDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )


def infer_feature_type_ids(table_cols: list[str]) -> tuple[list[int], dict[int, str]]:
    label_map = {
        0: "demographic",
        1: "genetic",
        2: "cognitive",
        3: "functional_psych",
        4: "other",
    }
    type_ids: list[int] = []
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


def build_denoiser(config: dict, table_num_features: int, feature_type_ids: list[int]) -> BSFETableJointPriorLatentBridgeDenoiser3D:
    model_cfg = config["model"]
    return BSFETableJointPriorLatentBridgeDenoiser3D(
        latent_channels=int(config["dvae"].get("latent_dim", 1)),
        cond_channels=int(model_cfg.get("cond_channels", 1)),
        table_num_features=int(table_num_features),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(model_cfg["channel_mults"]),
        time_dim=int(model_cfg["time_dim"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_groups=int(model_cfg.get("num_groups", 16)),
        use_attention=bool(model_cfg.get("use_attention", True)),
        attention_level=int(model_cfg.get("attention_level", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        table_token_dim=int(model_cfg.get("table_token_dim", 128)),
        table_cross_attn_heads=int(model_cfg.get("table_cross_attn_heads", 4)),
        table_joint_depth=int(model_cfg.get("table_joint_depth", 2)),
        mri_token_grid=tuple(model_cfg.get("mri_token_grid", [4, 6, 4])),
        reverse_pool_grid=tuple(model_cfg.get("reverse_pool_grid", [4, 4, 4])),
        num_feature_types=int(model_cfg.get("num_feature_types", 5)),
        feature_type_ids=feature_type_ids,
        max_timesteps=int(config["bridge"]["timesteps"]),
    )


def build_bridge(config: dict, model: torch.nn.Module) -> ResidualBridgeDiffusion:
    return ResidualBridgeDiffusion(
        model=model,
        timesteps=int(config["bridge"]["timesteps"]),
        bridge_scale=float(config["bridge"].get("bridge_scale", 1.0)),
        lambda_residual=float(config["loss"].get("lambda_residual", 1.0)),
        lambda_pet=float(config["loss"].get("lambda_pet", 1.0)),
    )


def encode_pet(dvae: torch.nn.Module, pet: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mu, _ = dvae.encoder(pet)
    return mu


def decode_pet(dvae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return dvae.decoder(latent)


@torch.no_grad()
def encode_bsfe_condition(bsfe: torch.nn.Module, head: torch.nn.Module, mri: torch.Tensor) -> torch.Tensor:
    features = bsfe(mri)
    outputs = head(features["pyramid"], features["global"], features["tokens"])
    return outputs["latent_prediction"]


def aggregate_rows(rows: list[dict]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    out: dict[str, float] = {"num_val_folds": float(len(df))}
    for key in ["mae", "rmse", "psnr", "ssim"]:
        out[f"{key}_mean"] = float(df[key].mean())
        out[f"{key}_std"] = float(df[key].std(ddof=0))
    return out


@torch.no_grad()
def evaluate_sampling(
    bridge: ResidualBridgeDiffusion,
    dvae: torch.nn.Module,
    bsfe: torch.nn.Module,
    bsfe_head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    bridge.eval()
    rows = []
    for batch in loader:
        mri = batch["input"].to(device)
        pet = batch["target"].to(device)
        brain_mask = batch["brain_mask"].to(device)
        table = batch["table"].to(device)
        table_mask = batch["table_mask"].to(device)
        pet_latent = encode_pet(dvae, pet)
        cond_latent = encode_bsfe_condition(bsfe, bsfe_head, mri)
        pred_latent = bridge.sample(
            shape=pet_latent.shape,
            condition=cond_latent,
            steps=steps,
            clip_prediction=False,
            model_kwargs={"table": table, "table_mask": table_mask},
        )
        pred = decode_pet(dvae, pred_latent)
        metrics = compute_image_metrics_detailed(pred, pet, mask=brain_mask, data_range=2.0)
        rows.append({k: float(v) for k, v in metrics.items() if k in {"mae", "rmse", "psnr", "ssim"}})
    df = pd.DataFrame(rows)
    return {key: float(df[key].mean()) for key in ["mae", "rmse", "psnr", "ssim"]}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() and str(config["training"].get("device", "cuda")) == "cuda" else "cpu")
    paths = prepare_experiment_dir(REPO_ROOT / config["experiment"]["output_dir"], config)

    train_dataset = build_dataset(config, "train")
    val_splits = list(config["evaluation"].get("val_splits", []))
    val_datasets = {
        split: build_dataset(
            config,
            split,
            tabular_stats=train_dataset.tabular_stats,
            table_cols=train_dataset.table_cols,
        )
        for split in val_splits
    }
    train_loader = build_loader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        drop_last=True,
    )
    val_loaders = {
        split: build_loader(
            dataset,
            batch_size=int(config["training"].get("batch_size", 1)),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 2)),
            pin_memory=bool(config["training"].get("pin_memory", True)),
            drop_last=False,
        )
        for split, dataset in val_datasets.items()
    }

    dvae = load_bsfe_dvae(config, device)
    latent_shape = infer_bsfe_latent_shape(dvae, tuple(config["data"]["image_size"]), device)
    bsfe, bsfe_head = build_bsfe_models(config, latent_shape, device)
    bsfe_ckpt = torch.load(REPO_ROOT / config["bsfe"]["checkpoint_path"], map_location=device)
    if isinstance(bsfe_ckpt, dict) and "bsfe" in bsfe_ckpt and "head" in bsfe_ckpt:
        bsfe.load_state_dict(bsfe_ckpt["bsfe"], strict=True)
        bsfe_head.load_state_dict(bsfe_ckpt["head"], strict=True)
    else:
        bsfe.load_state_dict(torch.load(REPO_ROOT / config["bsfe"]["checkpoint_path"], map_location=device), strict=True)
    bsfe.eval()
    bsfe_head.eval()
    for module in (bsfe, bsfe_head):
        for param in module.parameters():
            param.requires_grad_(False)

    feature_type_ids, feature_type_map = infer_feature_type_ids(train_dataset.table_cols)
    model = build_denoiser(config, len(train_dataset.table_cols), feature_type_ids).to(device)
    bridge = build_bridge(config, model).to(device)
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    train_csv = paths.root / "train_log.csv"
    val_csv = paths.root / "val_log.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss"])
    with val_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"])

    append_log(paths.train_log, f"Table feature types: {feature_type_map}")
    append_log(paths.train_log, f"Feature type ids: {dict(zip(train_dataset.table_cols, feature_type_ids))}")

    best_mae = float("inf")
    epochs = int(config["training"].get("epochs", 100))
    sample_interval = int(config["evaluation"].get("sample_val_interval", 1))
    for epoch in range(1, epochs + 1):
        bridge.train()
        loss_sum = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            mri = batch["input"].to(device)
            pet = batch["target"].to(device)
            table = batch["table"].to(device)
            table_mask = batch["table_mask"].to(device)
            pet_latent = encode_pet(dvae, pet)
            cond_latent = encode_bsfe_condition(bsfe, bsfe_head, mri)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = bridge.compute_training_losses(
                pet_latent,
                cond_latent,
                model_kwargs={"table": table, "table_mask": table_mask},
            )
            loss.backward()
            optimizer.step()
            bs = int(mri.shape[0])
            loss_sum += float(loss.item()) * bs
            seen += bs
        train_loss = loss_sum / max(seen, 1)
        with train_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss])

        log_row = {"epoch": epoch, "train_loss": train_loss}
        if sample_interval > 0 and epoch % sample_interval == 0:
            fold_rows = []
            for split, loader in val_loaders.items():
                metrics = evaluate_sampling(
                    bridge,
                    dvae,
                    bsfe,
                    bsfe_head,
                    loader,
                    device,
                    steps=int(config["evaluation"].get("sample_val_steps", 20)),
                )
                metrics["val_split"] = split
                fold_rows.append(metrics)
            pd.DataFrame(fold_rows).to_csv(paths.predictions / f"sampling_epoch_{epoch:04d}_fold_summary.csv", index=False)
            agg = aggregate_rows(fold_rows)
            with val_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch,
                    agg["mae_mean"], agg["mae_std"],
                    agg["rmse_mean"], agg["rmse_std"],
                    agg["psnr_mean"], agg["psnr_std"],
                    agg["ssim_mean"], agg["ssim_std"],
                ])
            log_row.update(agg)
            if agg["mae_mean"] < best_mae:
                best_mae = agg["mae_mean"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "config": config,
                        "table_cols": train_dataset.table_cols,
                        "feature_type_ids": feature_type_ids,
                    },
                    paths.checkpoints / "best_mae.pt",
                )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "table_cols": train_dataset.table_cols,
                "feature_type_ids": feature_type_ids,
            },
            paths.checkpoints / "last_model.pt",
        )
        append_log(paths.train_log, str(log_row))


if __name__ == "__main__":
    main()
