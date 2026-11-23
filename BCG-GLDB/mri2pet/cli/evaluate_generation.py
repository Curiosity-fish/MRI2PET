"""Generate PET, per-sample metrics and reusable predictions for an unseen target set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml


from mri2pet.metrics.image import ImageMetricCollection
from mri2pet.cli.train_generation import (
    build_ae,
    build_model,
    make_dataset,
    resolve,
    set_seed,
    synthesize,
    type_ids,
)


def run(source_config, checkpoint, target_config, splits, output):
    source = yaml.safe_load(Path(source_config).read_text(encoding="utf-8"))
    target = yaml.safe_load(Path(target_config).read_text(encoding="utf-8"))
    source_root = Path(source["data"]["root"]).expanduser().resolve()
    target_root = Path(target["data"]["root"]).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(source.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This source-training object fits its clinical statistics using source
    # subjects only. The target dataset only receives those fitted values.
    source_training = make_dataset(source, source_root, "train")
    if source["data"]["table_cols"] != target["data"]["table_cols"]:
        raise ValueError("Cross-database clinical field order is not harmonized")
    target_sets = [make_dataset(target, target_root, split, source_training) for split in splits]
    dataset = target_sets[0] if len(target_sets) == 1 else ConcatDataset(target_sets)
    loader = DataLoader(dataset, batch_size=int(source["training"]["batch_size"]), shuffle=False)
    ae = build_ae(source, source_root, device)
    model = build_model(source, type_ids(source_training.table_cols), device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    metric = ImageMetricCollection(data_range=2.0)
    results = []
    with torch.no_grad():
        for batch in loader:
            mri = batch["input"].to(device).float()
            pet = batch["target"].to(device).float()
            table = batch["table"].to(device).float()
            missing = batch["table_mask"].to(device).float()
            prediction = synthesize(model, ae, mri, table, missing, source)
            for index in range(len(mri)):
                sample_id = str(batch["sample_id"][index])
                subject_id = str(batch["subject_id"][index])
                path = prediction_dir / f"{sample_id}.pt"
                torch.save({"prediction": prediction[index].float().cpu()}, path)
                values = metric(prediction[index : index + 1], pet[index : index + 1])
                results.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "prediction_path": str(path),
                        **values,
                    }
                )
    with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "subject_id",
                "prediction_path",
                "psnr",
                "ssim",
                "mae",
                "rmse",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in results])),
            "std": (
                float(np.std([row[name] for row in results], ddof=1)) if len(results) > 1 else 0.0
            ),
        }
        for name in ("psnr", "ssim", "mae", "rmse")
    }
    summary["n"] = len(results)
    summary["target_splits"] = list(splits)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument(
        "--splits",
        default="final_val",
        help="Comma-separated target splits; use train,val,final_val for all target samples",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.source_config,
        args.checkpoint,
        args.target_config,
        tuple(part.strip() for part in args.splits.split(",")),
        args.output,
    )


if __name__ == "__main__":
    main()
