"""Five independent models on train folds, each tested on the same fixed 96 samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import yaml

from mri2pet.data.classification import (
    ClassificationDataset,
    fit_clinical_stats,
    load_index,
    read_clinical_table,
)
from mri2pet.metrics.classification import classify_metrics, five_model_summary
from mri2pet.models.classification import TGLM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_class_weights(frame):
    counts = np.bincount(frame.label.astype(int).to_numpy(), minlength=3)
    if np.any(counts == 0):
        raise ValueError("Every training fold must contain CN, MCI and AD")
    if np.all(counts == counts[0]):
        return None
    weights = len(frame) / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion, device):
    model.eval()
    rows = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            table = batch["table"].to(device)
            missing = batch["table_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(image, table, missing)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits.float(), labels)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            losses.extend([float(loss)] * len(labels))
            for index in range(len(labels)):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "subject_id": batch["subject_id"][index],
                        "label": int(labels[index]),
                        "p_cn": float(probabilities[index, 0]),
                        "p_mci": float(probabilities[index, 1]),
                        "p_ad": float(probabilities[index, 2]),
                    }
                )
    metrics = classify_metrics(
        [row["label"] for row in rows], [[row["p_cn"], row["p_mci"], row["p_ad"]] for row in rows]
    )
    return float(np.mean(losses)), metrics, rows


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "subject_id", "label", "p_cn", "p_mci", "p_ad"]
        )
        writer.writeheader()
        writer.writerows(rows)


def check_experiment(exp):
    required = {"model", "modality", "pet_path_col"}
    missing = required - set(exp)
    if missing:
        raise ValueError(f"Experiment lacks fields: {sorted(missing)}")
    if exp["model"] != "tglm" or exp["modality"] != "mri_pet":
        raise ValueError("This repository trains the full TGLM with MRI+PET+clinical input")


def run_experiment(cfg, name, exp, training_pool, fixed_test, clinical_table, device):
    check_experiment(exp)
    output_root = Path(cfg["output_dir"]).expanduser().resolve()
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    spatial = tuple(cfg["data"].get("image_shape", (160, 192, 160)))
    if spatial != (160, 192, 160):
        raise ValueError("The thesis uses 160×192×160 classification volumes")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    fold_file = cfg["data"].get("fold_split_csv")
    if fold_file:
        saved = pd.read_csv(data_root / fold_file, dtype={"subject_id": str, "sample_id": str})
        required = {"fold", "subject_id", "sample_id", "split"}
        if not required.issubset(saved.columns):
            raise ValueError(f"Saved fold file lacks columns: {sorted(required - set(saved.columns))}")
        sample_keys = list(zip(training_pool.subject_id.astype(str), training_pool.sample_id.astype(str)))
        test_keys = set(zip(fixed_test.subject_id.astype(str), fixed_test.sample_id.astype(str)))
        if set(saved["fold"].unique()) != {1, 2, 3, 4, 5}:
            raise ValueError("Saved fold file must contain exactly folds 1 to 5")
        fold_indices = []
        validation_counts = {key: 0 for key in sample_keys}
        for fold_number in range(1, 6):
            part = saved[saved["fold"] == fold_number]
            if part.duplicated(["subject_id", "sample_id"]).any():
                raise ValueError(f"Duplicate sample in saved fold {fold_number}")
            key_to_split = {(row.subject_id, row.sample_id): row.split for row in part.itertuples()}
            if set(key_to_split) != set(sample_keys) | test_keys:
                raise ValueError(f"Saved fold {fold_number} does not match current pool/test")
            if any(key_to_split[k] != "final_val" for k in test_keys):
                raise ValueError(f"Saved fold {fold_number} changes the fixed test")
            if any(key_to_split[k] not in {"train", "val"} for k in sample_keys):
                raise ValueError(f"Invalid pool assignment in saved fold {fold_number}")
            tr = [i for i,k in enumerate(sample_keys) if key_to_split[k] == "train"]
            va = [i for i,k in enumerate(sample_keys) if key_to_split[k] == "val"]
            for i in va:
                validation_counts[sample_keys[i]] += 1
            fold_indices.append((tr, va))
        if any(n != 1 for n in validation_counts.values()):
            raise ValueError("Each training-pool sample must be in validation exactly once")
    else:
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_indices = folds.split(training_pool, training_pool.label, training_pool.subject_id)
    results = []
    fixed_ids = fixed_test.sample_id.astype(str).tolist()
    for fold, (train_indices, val_indices) in enumerate(fold_indices, start=1):
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        train_frame = training_pool.iloc[train_indices].copy()
        val_frame = training_pool.iloc[val_indices].copy()
        if set(train_frame.subject_id.astype(str)) & set(val_frame.subject_id.astype(str)):
            raise ValueError("A subject occurs in both current-fold train and validation")
        stats = fit_clinical_stats(train_frame, clinical_table)
        np.savez(
            fold_dir / "training_clinical_stats.npz",
            median=stats["median"],
            mean=stats["mean"],
            std=stats["std"],
        )
        (fold_dir / "train_sample_ids.json").write_text(
            json.dumps(stats["fit_sample_ids"], indent=2), encoding="utf-8"
        )
        datasets = {
            "train": ClassificationDataset(
                train_frame,
                clinical_table,
                stats,
                data_root,
                modality=exp["modality"],
                pet_path_col=exp["pet_path_col"],
                spatial=spatial,
            ),
            "val": ClassificationDataset(
                val_frame,
                clinical_table,
                stats,
                data_root,
                modality=exp["modality"],
                pet_path_col=exp["pet_path_col"],
                spatial=spatial,
            ),
            "test": ClassificationDataset(
                fixed_test,
                clinical_table,
                stats,
                data_root,
                modality=exp["modality"],
                pet_path_col=exp["pet_path_col"],
                spatial=spatial,
            ),
        }
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=4,
                shuffle=split == "train",
                num_workers=int(cfg["training"].get("num_workers", 0)),
                pin_memory=device.type == "cuda",
            )
            for split, dataset in datasets.items()
        }
        # One study seed; consecutive folds construct independent model instances.
        channels = 2 if exp["modality"] == "mri_pet" else 1
        model = TGLM(channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
        )
        class_weights = estimate_class_weights(train_frame)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        best_f1 = -float("inf")
        for epoch in range(1, 101):
            model.train()
            epoch_losses = []
            for batch in loaders["train"]:
                image = batch["image"].to(device)
                table = batch["table"].to(device)
                missing = batch["table_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    logits = model(image, table, missing)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = criterion(logits.float(), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            _, val_result, _ = evaluate(model, loaders["val"], criterion, device)
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation": val_result,
            }
            with (fold_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if val_result["macro_f1"] > best_f1:
                best_f1 = val_result["macro_f1"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_name": exp["model"],
                        "modality": exp["modality"],
                        "epoch": epoch,
                        "validation": val_result,
                        "config": cfg,
                    },
                    fold_dir / "best.pt",
                )
        checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        _, test_result, rows = evaluate(model, loaders["test"], criterion, device)
        if [row["sample_id"] for row in rows] != fixed_ids:
            raise AssertionError("The fixed test subjects/order changed across folds")
        save_rows(fold_dir / "fixed_test_predictions.csv", rows)
        (fold_dir / "fixed_test_metrics.json").write_text(
            json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(test_result)
        print(
            json.dumps({"experiment": name, "fold": fold, "test": test_result}, ensure_ascii=False),
            flush=True,
        )
    summary = five_model_summary(results)
    (output / "five_independent_models_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run(config_path, selected=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if int(cfg["training"]["epochs"]) != 100 or int(cfg["training"]["batch_size"]) != 4:
        raise ValueError("The thesis specifies 100 epochs and batch size 4")
    if float(cfg["training"]["lr"]) != 1e-4 or int(cfg.get("seed", 42)) != 42:
        raise ValueError("The thesis specifies AdamW learning rate 1e-4 and seed 42")
    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    training_pool, fixed_test = load_index(
        data_root / cfg["data"]["labels_csv"],
        data_root / cfg["data"]["split_csv"],
        train_splits=tuple(cfg["data"].get("train_splits", ["train", "val"])),
        test_split=cfg["data"].get("test_split", "final_val"),
    )
    generation_split_name = cfg["data"].get("generation_split_csv")
    if not generation_split_name:
        raise ValueError(
            "Supply the generation split CSV to verify the fixed test was unseen by BCG-GLDB"
        )
    generation_split = pd.read_csv(data_root / generation_split_name)
    if not {"subject_id", "split"}.issubset(generation_split.columns):
        raise ValueError("Generation split CSV needs subject_id and split columns")
    seen_by_generator = set(
        generation_split.loc[generation_split["split"].isin(("train", "val")), "subject_id"].astype(
            str
        )
    )
    leakage = seen_by_generator & set(fixed_test.subject_id.astype(str))
    if leakage:
        raise ValueError(
            f"The fixed classification test includes generation train/val subjects: {sorted(leakage)[:10]}"
        )
    clinical_table = read_clinical_table(data_root / cfg["data"]["table_csv"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = cfg["experiments"]
    names = [selected] if selected else list(experiments)
    for name in names:
        if name not in experiments:
            raise KeyError(f"Unknown experiment: {name}")
        experiment = experiments[name]
        run_experiment(cfg, name, experiment, training_pool, fixed_test, clinical_table, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", help="Select the real- or synthetic-PET TGLM input")
    args = parser.parse_args()
    run(args.config, args.experiment)


if __name__ == "__main__":
    main()
