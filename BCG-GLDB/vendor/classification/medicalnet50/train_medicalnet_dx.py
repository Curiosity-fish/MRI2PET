from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

# Support direct script execution without requiring manual PYTHONPATH setup.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MRI2PET.classier import (
    DiaMondMRIClassifier,
    PairedMRIPETDiagnosisDataset,
    PairedPETDiagnosisDataset,
    PairedT1DiagnosisDataset,
    create_diamond_mri_classifier,
    create_medicalnet_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedicalNet for ADNI 3-class MRI diagnosis classification.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[list[float]],
    num_classes: int,
) -> dict[str, Any]:
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(num_classes))
    accuracy = correct / total if total else 0.0

    recalls = []
    precisions = []
    f1s = []
    for i in range(num_classes):
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(num_classes) if j != i)
        fp = sum(cm[j][i] for j in range(num_classes) if j != i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    auc = None
    try:
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_prob_np = np.asarray(y_prob, dtype=np.float32)
        auc = float(roc_auc_score(y_true_np, y_prob_np, multi_class="ovr", average="macro"))
    except Exception:
        auc = None

    return {
        "accuracy": accuracy,
        "macro_pre": float(sum(precisions) / num_classes),
        "macro_rec": float(sum(recalls) / num_classes),
        "macro_f1": float(sum(f1s) / num_classes),
        "macro_auc": auc,
        "balanced_accuracy": float(sum(recalls) / num_classes),
        "macro_precision": float(sum(precisions) / num_classes),
        "macro_recall": float(sum(recalls) / num_classes),
        "confusion_matrix": cm,
    }


def create_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, Any]


class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    logger: Logger,
    optimizer=None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    phase = "train" if training else "val"

    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    num_classes: int | None = None

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{total_epochs:03d} [{phase}]",
        leave=False,
        dynamic_ncols=True,
    )
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in progress:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            num_classes = int(logits.shape[-1])
            loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            y_prob.extend(prob.detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    if num_classes is None:
        raise ValueError(f"No batches were processed during {phase}.")
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    logger.log(
        f"Epoch {epoch:03d} {phase} | "
        f"loss={total_loss / max(total_count, 1):.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"pre={metrics['macro_pre']:.4f} "
        f"rec={metrics['macro_rec']:.4f} "
        f"f1={metrics['macro_f1']:.4f} "
        f"auc={metrics['macro_auc'] if metrics['macro_auc'] is not None else 'NA'}"
    )
    return EpochResult(loss=total_loss / max(total_count, 1), metrics=metrics)


def summarize_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_macro_pre": metrics["macro_pre"],
        f"{prefix}_macro_rec": metrics["macro_rec"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_macro_auc": metrics["macro_auc"],
        f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
        f"{prefix}_confusion_matrix": metrics["confusion_matrix"],
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    experiment_name = config["experiment"]["name"]
    output_dir = Path(config["experiment"]["output_dir"]).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    predictions_dir = output_dir / "predictions"
    log_path = output_dir / "train.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    logger = Logger(log_path)

    seed = int(config["train"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"].get("device", "cuda") == "cuda" else "cpu")

    data_cfg = config["data"]
    split_csv = data_cfg["split_csv"]
    labels_csv = data_cfg["labels_csv"]
    spatial_size = tuple(int(v) for v in data_cfg["spatial_size"])
    fold = data_cfg.get("fold")
    if fold == "" or fold == -1:
        fold = None
    input_modality = str(data_cfg.get("input_modality", "mri")).lower()

    if input_modality == "mri":
        dataset_cls = PairedT1DiagnosisDataset
    elif input_modality == "pet":
        dataset_cls = PairedPETDiagnosisDataset
    elif input_modality == "mri_pet":
        dataset_cls = PairedMRIPETDiagnosisDataset
    else:
        raise ValueError(f"Unsupported input_modality: {input_modality}")

    train_ds = dataset_cls(
        labels_csv=labels_csv,
        split_csv=split_csv,
        split="train",
        fold=fold,
        spatial_size=spatial_size,
        cache_dir=data_cfg.get("cache_dir"),
    )
    val_ds = dataset_cls(
        labels_csv=labels_csv,
        split_csv=split_csv,
        split="val",
        fold=fold,
        spatial_size=spatial_size,
        cache_dir=data_cfg.get("cache_dir"),
    )
    final_val_ds = dataset_cls(
        labels_csv=labels_csv,
        split_csv=split_csv,
        split="final_val",
        fold=None,
        spatial_size=spatial_size,
        cache_dir=data_cfg.get("cache_dir"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
    )
    final_val_loader = DataLoader(
        final_val_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
    )

    model_cfg = config["model"]
    model_name = str(model_cfg.get("name", "medicalnet")).lower()
    if model_name == "medicalnet":
        model = create_medicalnet_classifier(
            model_depth=int(model_cfg["depth"]),
            n_classes=int(model_cfg["num_classes"]),
            in_channels=int(model_cfg.get("in_channels", 1)),
            shortcut_type=str(model_cfg.get("shortcut_type", "A")),
            dropout=float(model_cfg.get("dropout", 0.0)),
            head_hidden_dims=[int(v) for v in model_cfg.get("head_hidden_dims", [])],
            checkpoint_path=model_cfg.get("pretrained_path") or None,
        ).to(device)
    elif model_name == "diamond":
        model = create_diamond_mri_classifier(
            image_size=int(model_cfg["image_size"]),
            block_size=int(model_cfg["block_size"]),
            patch_size=int(model_cfg["patch_size"]),
            num_classes=int(model_cfg["num_classes"]),
            in_channels=int(model_cfg.get("in_channels", 1)),
            dim=int(model_cfg["dim"]),
            depth=int(model_cfg["depth"]),
            heads=int(model_cfg["heads"]),
            mlp_dim=int(model_cfg["mlp_dim"]),
            head_hidden_dim1=int(model_cfg.get("head_hidden_dim1", 192)),
            head_hidden_dim2=int(model_cfg.get("head_hidden_dim2", 64)),
            pool=str(model_cfg.get("pool", "cls")),
            dim_head=int(model_cfg.get("dim_head", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            emb_dropout=float(model_cfg.get("emb_dropout", 0.0)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model.name: {model_name}")

    class_weight = config["train"].get("class_weight")
    if class_weight:
        weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        config["train"]["optimizer"],
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    resume_path = config.get("train", {}).get("resume_from")
    resume_epoch = 0
    if resume_path:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        resume_epoch = int(resume_ckpt.get("epoch", 0))
        logger.log(f"Resumed from {resume_path} at epoch {resume_epoch}")

    monitor = str(config["eval"].get("monitor", "macro_f1"))
    best_value = float("-inf")
    best_epoch = -1
    best_val_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    # A resumed run must compete with the best checkpoint from before the
    # interruption, rather than treating the first resumed epoch as best.
    prior_best_path = checkpoints_dir / "best.pt"
    if resume_path and prior_best_path.exists():
        prior_best = torch.load(prior_best_path, map_location=device, weights_only=False)
        best_value = float(prior_best.get("best_value", float("-inf")))
        best_epoch = int(prior_best.get("epoch", -1))

    logger.log(f"Experiment: {experiment_name}")
    logger.log(f"Output dir: {output_dir}")
    logger.log(f"Device: {device}")
    logger.log(f"Model: {model_name}")
    logger.log(f"Input modality: {input_modality}")
    logger.log(f"Labels CSV: {labels_csv}")
    logger.log(f"Split CSV: {split_csv}")
    logger.log(f"Fold: {fold if fold is not None else 'none'}")
    logger.log(
        f"Train samples: {len(train_ds)} | "
        f"Val samples: {len(val_ds)} | "
        f"Final val samples: {len(final_val_ds)}"
    )

    total_epochs = int(config["train"]["epochs"])
    for epoch in range(resume_epoch + 1, total_epochs + 1):
        train_res = run_epoch(model, train_loader, criterion, device, epoch, total_epochs, logger, optimizer=optimizer)
        val_res = run_epoch(model, val_loader, criterion, device, epoch, total_epochs, logger, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_res.loss,
            "val_loss": val_res.loss,
            **{f"train_{k}": v for k, v in train_res.metrics.items() if k != "confusion_matrix"},
            **{f"val_{k}": v for k, v in val_res.metrics.items() if k != "confusion_matrix"},
        }
        history.append(row)

        current = float(val_res.metrics[monitor])
        is_best = current > best_value
        if is_best:
            best_value = current
            best_epoch = epoch
            best_val_metrics = dict(val_res.metrics)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "monitor": monitor,
                    "best_value": best_value,
                    "config": config,
                },
                checkpoints_dir / "best.pt",
            )
            (predictions_dir / "best_val_confusion_matrix.json").write_text(
                json.dumps(val_res.metrics["confusion_matrix"], indent=2),
                encoding="utf-8",
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            checkpoints_dir / "last.pt",
        )

        logger.log(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_res.loss:.4f} val_loss={val_res.loss:.4f} "
            f"train_acc={train_res.metrics['accuracy']:.4f} "
            f"train_pre={train_res.metrics['macro_pre']:.4f} "
            f"train_rec={train_res.metrics['macro_rec']:.4f} "
            f"train_f1={train_res.metrics['macro_f1']:.4f} "
            f"train_auc={train_res.metrics['macro_auc'] if train_res.metrics['macro_auc'] is not None else 'NA'} "
            f"val_acc={val_res.metrics['accuracy']:.4f} "
            f"val_pre={val_res.metrics['macro_pre']:.4f} "
            f"val_rec={val_res.metrics['macro_rec']:.4f} "
            f"val_f1={val_res.metrics['macro_f1']:.4f} "
            f"val_auc={val_res.metrics['macro_auc'] if val_res.metrics['macro_auc'] is not None else 'NA'}"
        )

    best_final_val_metrics: dict[str, Any] | None = None
    if best_epoch >= 0:
        best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_val_res = run_epoch(model, final_val_loader, criterion, device, best_epoch, total_epochs, logger, optimizer=None)
        best_final_val_metrics = dict(final_val_res.metrics)
        (predictions_dir / "best_final_val_confusion_matrix.json").write_text(
            json.dumps(final_val_res.metrics["confusion_matrix"], indent=2),
            encoding="utf-8",
        )
        logger.log(
            "Best model final_val metrics | "
            f"epoch={best_epoch:03d} "
            f"acc={final_val_res.metrics['accuracy']:.4f} "
            f"pre={final_val_res.metrics['macro_pre']:.4f} "
            f"rec={final_val_res.metrics['macro_rec']:.4f} "
            f"f1={final_val_res.metrics['macro_f1']:.4f} "
            f"auc={final_val_res.metrics['macro_auc'] if final_val_res.metrics['macro_auc'] is not None else 'NA'}"
        )

    final_metrics = {
        "experiment": experiment_name,
        "best_monitor": monitor,
        "best_value": best_value,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_final_val_metrics": best_final_val_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    if best_val_metrics is not None:
        best_summary = {
            "experiment": experiment_name,
            "best_epoch": best_epoch,
            "best_monitor": monitor,
            "best_value": best_value,
            **summarize_metrics("val", best_val_metrics),
        }
        if best_final_val_metrics is not None:
            best_summary.update(summarize_metrics("final_val", best_final_val_metrics))
        (output_dir / "best_eval_metrics.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
        logger.log(
            "Best val metrics | "
            f"epoch={best_epoch:03d} "
            f"monitor={monitor} "
            f"value={best_value:.4f} "
            f"acc={best_val_metrics['accuracy']:.4f} "
            f"pre={best_val_metrics['macro_pre']:.4f} "
            f"rec={best_val_metrics['macro_rec']:.4f} "
            f"f1={best_val_metrics['macro_f1']:.4f} "
            f"auc={best_val_metrics['macro_auc'] if best_val_metrics['macro_auc'] is not None else 'NA'}"
        )

    notes = output_dir / "notes.md"
    if not notes.exists():
        notes.write_text(
            "\n".join(
                [
                    f"# {experiment_name}",
                    "",
                    "## Summary",
                    "",
                    "- Backbone: MedicalNet",
                    "- Input: T1 MRI",
                    "- Task: 3-class diagnosis classification",
                    "",
                    "## Notes",
                    "",
                    "- Fill in conclusions after training finishes.",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
