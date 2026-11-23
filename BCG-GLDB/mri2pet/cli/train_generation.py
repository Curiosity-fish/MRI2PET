"""Stage 2: subject-isolated stochastic BCG-GLDB training and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml


from mri2pet.data.paired_mri_pet_clinical import PairedT1FDGTableDataset
from mri2pet.metrics.image import ImageMetricCollection
from mri2pet.models.generation import PaperBCGGLDB, SharedPatchAE
from mri2pet.losses.generation import (
    bridge_state,
    generation_loss,
    reverse_step,
    validate_parc98_labelmap,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def validate_thesis_config(cfg):
    d, t, bridge, train = cfg["data"], cfg["training"], cfg["bridge"], cfg["model"]
    expected = [
        (tuple(d["image_shape"]), (160, 192, 160), "image_shape"),
        (tuple(d["patch_shape"]), (40, 48, 40), "patch_shape"),
        (tuple(train["channels"]), (96, 192, 384, 768), "U-Net channels"),
        (tuple(train["mri_channels"]), (48, 96, 192, 384), "MRI channels"),
        (int(train["time_dim"]), 384, "time_dim"),
        (int(train["global_depth"]), 2, "global_depth"),
        (int(train["global_heads"]), 4, "global_heads"),
        (int(bridge["steps"]), 10, "bridge steps"),
        (float(bridge["sigma_max"]), 0.25, "sigma_max"),
        (float(bridge["eta"]), 1.0, "stochastic eta"),
        (int(t["batch_size"]), 2, "physical batch_size"),
        (int(t["gradient_accumulation"]), 1, "gradient_accumulation"),
        (int(t["epochs"]), 300, "epochs"),
        (float(t["lr"]), 1e-4, "learning rate"),
        (float(t["weight_decay"]), 1e-4, "weight decay"),
    ]
    for actual, required, name in expected:
        if actual != required:
            raise ValueError(f"Thesis requires {name}={required}, received {actual}")


def load_atlas(path: Path, image_shape, device):
    if not path.exists():
        raise FileNotFoundError(f"Parc98 labelmap not found: {path}")
    array = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if array.shape[1] == image_shape[1] + 4:
        array = array[:, 2:-2, :]
    labelmap = torch.from_numpy(array.copy())[None, None].to(device)
    if tuple(labelmap.shape[-3:]) != tuple(image_shape):
        labelmap = F.interpolate(labelmap, size=image_shape, mode="nearest")
    labelmap = labelmap[0, 0].round().long()
    validate_parc98_labelmap(labelmap)
    return labelmap


def make_dataset(cfg, root: Path, split: str, train=None):
    data = cfg["data"]
    return PairedT1FDGTableDataset(
        pairs_csv=resolve(root, data["pairs_csv"]),
        split_csv=resolve(root, data["split_csv"]),
        table_csv=resolve(root, data["table_csv"]),
        split=split,
        fold=None,
        spatial_size=tuple(data["image_shape"]),
        tabular_stats=None if train is None else train.tabular_stats,
        table_cols=list(data["table_cols"]),
        max_table_delta_days=90,
        cache_dir=resolve(root, data["cache_dir"]) if data.get("cache_dir") else None,
    )


def assert_isolated(datasets):
    groups = {name: set(ds.df.subject_id.astype(str)) for name, ds in datasets.items()}
    for left, a in groups.items():
        for right, b in groups.items():
            if left < right and a & b:
                raise ValueError(f"Subjects overlap between {left} and {right}")


def build_ae(cfg, data_root, device):
    ae_cfg = cfg["patch_ae"]
    ae = SharedPatchAE(
        image_shape=cfg["data"]["image_shape"],
        patch_shape=cfg["data"]["patch_shape"],
        latent_channels=4,
        base_channels=int(ae_cfg.get("base_channels", 16)),
        chunk_size=int(ae_cfg.get("chunk_size", 16)),
    ).to(device)
    checkpoint = Path(ae_cfg["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        local_checkpoint = Path.cwd() / checkpoint
        checkpoint = local_checkpoint if local_checkpoint.is_file() else data_root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Train Stage 1 first; Patch-AE checkpoint is missing: {checkpoint}"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ae.load_state_dict(state["model"], strict=True)
    ae.freeze_for_bridge()
    return ae


def build_model(cfg, table_features, device):
    m = cfg["model"]
    model = PaperBCGGLDB(
        table_feature_types=table_features,
        channels=tuple(m["channels"]),
        mri_channels=tuple(m["mri_channels"]),
        time_dim=int(m["time_dim"]),
        token_dim=int(m.get("table_token_dim", 128)),
        global_token_dim=int(m.get("global_token_dim", 256)),
        global_depth=int(m["global_depth"]),
        global_heads=int(m["global_heads"]),
        bottleneck_shape=(5, 6, 5),
        use_mri_structure=bool(m.get("use_mri_structure", True)),
        use_clinical=bool(m.get("use_clinical", True)),
        use_global=bool(m.get("use_global", True)),
    ).to(device)
    return model


def type_ids(table_cols):
    # The fixed 25-field input uses demographics(0), genetics/family(1),
    # cognitive scores(2), and histories/vitals/examination(4).
    if len(table_cols) != 25:
        raise ValueError("The thesis specifies 25 harmonized clinical fields")
    return [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2] + [4] * 14


def mri_endpoint(mri):
    return F.avg_pool3d(mri, kernel_size=4, stride=4).repeat(1, 4, 1, 1, 1)


@torch.no_grad()
def synthesize(model, ae, mri, table, table_mask, cfg):
    steps = int(cfg["bridge"]["steps"])
    sigma_max = float(cfg["bridge"]["sigma_max"])
    eta = float(cfg["bridge"]["eta"])
    state = mri_endpoint(mri)
    for k in range(steps, 0, -1):
        time = torch.full((mri.shape[0],), k / float(steps), device=mri.device)
        residual = model(state, time, mri, table, table_mask)
        state = reverse_step(state, residual, k, steps, sigma_max, eta)
    return ae.decode_volume(state)


def score_dataset(model, ae, loader, cfg, device, output_csv):
    model.eval()
    ae.eval()
    metrics = ImageMetricCollection(data_range=2.0)
    rows = []
    with torch.no_grad():
        for batch in loader:
            mri = batch["input"].to(device).float()
            pet = batch["target"].to(device).float()
            table = batch["table"].to(device).float()
            table_mask = batch["table_mask"].to(device).float()
            prediction = synthesize(model, ae, mri, table, table_mask, cfg)
            for index in range(mri.shape[0]):
                values = metrics(prediction[index : index + 1], pet[index : index + 1])
                rows.append(
                    {
                        "subject_id": str(batch["subject_id"][index]),
                        "sample_id": str(batch["sample_id"][index]),
                        **values,
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("subject_id", "sample_id", "psnr", "ssim", "mae", "rmse")
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "std": float(np.std([row[name] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for name in ("psnr", "ssim", "mae", "rmse")
    }


def run(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_thesis_config(cfg)
    set_seed(int(cfg.get("seed", 42)))
    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    output = Path(cfg["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {"train": make_dataset(cfg, data_root, "train")}
    for split in ("val", "final_val"):
        datasets[split] = make_dataset(cfg, data_root, split, datasets["train"])
    assert_isolated(datasets)
    loaders = {
        name: DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=name == "train",
            num_workers=int(cfg["training"].get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        for name, ds in datasets.items()
    }
    ae = build_ae(cfg, data_root, device)
    model = build_model(cfg, type_ids(datasets["train"].table_cols), device)
    atlas = load_atlas(
        resolve(data_root, cfg["data"]["parc98_labelmap"]),
        tuple(cfg["data"]["image_shape"]),
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_ssim = -float("inf")
    for epoch in range(1, 301):
        model.train()
        accum = {name: 0.0 for name in ("total", "bridge", "image", "roi")}
        count = 0
        for batch in loaders["train"]:
            pet = batch["target"].to(device).float()
            mri = batch["input"].to(device).float()
            table = batch["table"].to(device).float()
            missing = batch["table_mask"].to(device).float()
            brain = batch["brain_mask"].to(device).float()
            with torch.no_grad():
                z_pet = ae.encode_volume(pet)
            z_mri = mri_endpoint(mri)
            k = torch.randint(1, 11, (pet.shape[0],), device=device)
            z_k = bridge_state(z_pet, z_mri, k, 10, 0.25)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                predicted_residual = model(z_k, k.float() / 10.0, mri, table, missing)
                reconstructed = ae.decode_volume(z_k - predicted_residual)
                losses = generation_loss(
                    predicted_residual,
                    z_k - z_pet,
                    z_pet,
                    k,
                    10,
                    0.25,
                    reconstructed,
                    pet,
                    brain,
                    atlas,
                    use_roi=bool(cfg["model"].get("use_roi", True)),
                )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for key in accum:
                accum[key] += float(losses[key].detach()) * pet.shape[0]
            count += pet.shape[0]
        train_result = {key: value / count for key, value in accum.items()}
        val_result = score_dataset(
            model, ae, loaders["val"], cfg, device, output / f"val_epoch_{epoch:03}.csv"
        )
        summary = {"epoch": epoch, "train": train_result, "val": val_result}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if val_result["ssim"]["mean"] > best_ssim:
            best_ssim = val_result["ssim"]["mean"]
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val_result},
                output / "best_generation.pt",
            )
            (output / "best_val.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    checkpoint = torch.load(output / "best_generation.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_result = score_dataset(
        model, ae, loaders["final_val"], cfg, device, output / "independent_test_per_sample.csv"
    )
    (output / "independent_test_summary.json").write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
