"""Stage 1: train the thesis Patch-AE with one equivalent L1 objective."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml


from mri2pet.data.paired_pet import PairedFDGPETDataset
from mri2pet.models.generation import SharedPatchAE


def run(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    output = Path(cfg["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = tuple(cfg["data"]["image_shape"])
    patch_shape = tuple(cfg["data"]["patch_shape"])
    ae = SharedPatchAE(
        image_shape=image_shape,
        patch_shape=patch_shape,
        latent_channels=int(cfg["model"]["latent_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        chunk_size=int(cfg["training"].get("patch_chunk_size", 16)),
    ).to(device)
    if float(cfg["training"].get("lambda_rec", 2.0)) != 2.0:
        raise ValueError("The thesis equivalent single-L1 reconstruction weight is 2.0")
    datasets = {
        split: PairedFDGPETDataset(
            pairs_csv=data_root / cfg["data"]["pairs_csv"],
            split_csv=data_root / cfg["data"]["split_csv"],
            split=split,
            fold=None,
            spatial_size=image_shape,
            normalize_targets=False,
        )
        for split in ("train", "val")
    }
    if set(datasets["train"].df.subject_id.astype(str)) & set(
        datasets["val"].df.subject_id.astype(str)
    ):
        raise ValueError("AE training and validation subjects overlap")
    loaders = {
        name: DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=name == "train",
            num_workers=int(cfg["training"].get("num_workers", 0)),
        )
        for name, ds in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        ae.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    history = []
    best = float("inf")
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        epoch_result = {"epoch": epoch}
        for split in ("train", "val"):
            ae.train(split == "train")
            total, count = 0.0, 0
            for batch in loaders[split]:
                pet = batch["target"].to(device)
                with torch.set_grad_enabled(split == "train"):
                    reconstruction = ae(pet)
                    loss = 2.0 * F.l1_loss(reconstruction, pet)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach()) * pet.shape[0]
                count += pet.shape[0]
            epoch_result[split + "_loss"] = total / count
        history.append(epoch_result)
        if epoch_result["val_loss"] < best:
            best = epoch_result["val_loss"]
            torch.save(
                {"model": ae.state_dict(), "config": cfg, "epoch": epoch},
                output / "best_patch_ae.pt",
            )
        print(json.dumps(epoch_result), flush=True)
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
