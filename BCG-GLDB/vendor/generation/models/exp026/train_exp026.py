from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.mri_pet_table_loader import MRI2PETTableDataset, safe_collate
from exp026.models.exp026_model import build_exp026_model
from exp026.training.trainer import Exp026Trainer
from exp026.utils.config import STAGE_NAMES, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the seven-stage exp026 MRI-to-PET model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_NAMES, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config: dict, split: str, train_dataset=None) -> MRI2PETTableDataset:
    data = config["data"]
    data_path = data["train_data"] if split == "train" else data.get("val_data")
    if not data_path:
        raise ValueError(f"data.{split}_data is not configured")
    return MRI2PETTableDataset(
        data_path=data_path,
        table_csv=data["table_csv"],
        desired_shape=tuple(int(v) for v in data["image_size"]),
        table_cols=(train_dataset.table_cols if train_dataset is not None else data.get("table_cols")),
        add_missing_mask=True,
        standardize_table=bool(data.get("standardize_table", True)),
        tabular_stats=(train_dataset.tab_stats if train_dataset is not None else None),
        allow_unmatched_table=bool(data.get("allow_unmatched_table", False)),
        match_policy=str(data.get("match_policy", "nearest")),
        max_date_delta_days=int(data.get("max_date_delta_days", 30)),
    )


def build_loader(dataset, config: dict, shuffle: bool) -> DataLoader:
    training = config["training"]
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=bool(training.get("pin_memory", True)),
        drop_last=bool(shuffle and training.get("drop_last", True)),
        collate_fn=safe_collate,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 2026)))
    requested_device = str(config["training"].get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is false")
    device = torch.device(requested_device)

    train_dataset = build_dataset(config, "train")
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = None
    if config["data"].get("val_data"):
        val_dataset = build_dataset(config, "val", train_dataset=train_dataset)
        val_loader = build_loader(val_dataset, config, shuffle=False)

    table_config = config["model"]["table"]
    feature_type_ids = table_config.get("feature_type_ids")
    if feature_type_ids is None:
        feature_type_ids = [0] * len(train_dataset.table_cols)
    if len(feature_type_ids) != len(train_dataset.table_cols):
        raise ValueError("model.table.feature_type_ids must match the selected table columns")
    model = build_exp026_model(
        config,
        table_num_features=len(train_dataset.table_cols),
        feature_type_ids=feature_type_ids,
    ).to(device)

    output_root = Path(config["experiment"]["output_dir"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (output_root / "table_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(train_dataset.tab_stats.to_dict(), handle, indent=2, ensure_ascii=False)

    trainer = Exp026Trainer(model, train_loader, val_loader, config, device)
    trainer.run(only_stage=args.stage)


if __name__ == "__main__":
    main()
