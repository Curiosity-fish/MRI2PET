"""Train-only voxel priors used by Exp027.

Atlas and residual priors are fitted from the training split only.  Cache reuse is
guarded by sample ids, geometry, preprocessing inputs, and (for residual priors)
the exact structural-decoder checkpoint that produced the source predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from exp027.data import resolve_path


def _dataset_ids(dataset: Any) -> list[str]:
    if hasattr(dataset, "df"):
        frame = dataset.df
        for column in ("sample_id", "subject_id", "id"):
            if column in frame.columns:
                return [str(value) for value in frame[column].tolist()]
    return [str(index) for index in range(len(dataset))]


def _prior_path(config: Dict[str, Any], repo_root: Path, key: str, default_name: str) -> Path:
    value = config.get("data", {}).get(key, default_name)
    path = resolve_path(repo_root, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _preprocess_signature(config: Dict[str, Any], repo_root: Path, dataset: Any) -> dict:
    data = config["data"]
    path_keys = ("pairs_csv", "split_csv", "table_csv", "parc98_labelmap_path")
    signature = {
        "image_size": [int(value) for value in data["image_size"]],
        "patch_size": [int(value) for value in data["patch_size"]],
        "standardize_table": bool(data.get("standardize_table", True)),
        "table_cols": [str(value) for value in getattr(dataset, "table_cols", [])],
    }
    for key in path_keys:
        value = data.get(key)
        signature[key] = str(resolve_path(repo_root, value)) if value is not None else None
    return signature


def _checkpoint_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _compatible(
    payload: Dict[str, Any],
    sample_ids: list[str],
    image_size: Iterable[int],
    signature: Optional[dict] = None,
    source_checkpoint: Optional[Path] = None,
) -> bool:
    if [str(value) for value in payload.get("sample_ids", [])] != sample_ids:
        return False
    if tuple(payload.get("image_size", ())) != tuple(int(v) for v in image_size):
        return False
    if signature is not None and payload.get("preprocess_signature") != signature:
        return False
    if source_checkpoint is not None:
        if not source_checkpoint.exists():
            return False
        if str(Path(payload.get("source_checkpoint", "")).resolve()) != str(source_checkpoint.resolve()):
            return False
        if payload.get("source_checkpoint_fingerprint") != _checkpoint_fingerprint(source_checkpoint):
            return False
    return True


@torch.no_grad()
def build_or_load_atlas_prior(
    config: Dict[str, Any],
    repo_root: Path,
    train_dataset: Any,
    *,
    rebuild: bool = False,
) -> Dict[str, Any]:
    image_size = tuple(int(v) for v in config["data"]["image_size"])
    sample_ids = _dataset_ids(train_dataset)
    signature = _preprocess_signature(config, repo_root, train_dataset)
    path = _prior_path(config, repo_root, "atlas_prior_path", "data/cache/exp027_voxel_atlas_train.pt")
    if path.exists() and not rebuild:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if _compatible(payload, sample_ids, image_size, signature):
            return payload

    sum_x = torch.zeros((1, 1, *image_size), dtype=torch.float64)
    sum_y = torch.zeros_like(sum_x)
    sum_xx = torch.zeros_like(sum_x)
    sum_xy = torch.zeros_like(sum_x)
    count = 0
    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        x = sample["image"].float().unsqueeze(0)
        y = sample["label"].float().unsqueeze(0)
        sum_x += x.double()
        sum_y += y.double()
        sum_xx += x.double().square()
        sum_xy += x.double() * y.double()
        count += 1
    if count == 0:
        raise ValueError("Cannot fit Exp027 atlas from an empty training dataset")

    mean_x = sum_x / count
    mean_y = sum_y / count
    covariance = sum_xy / count - mean_x * mean_y
    variance = (sum_xx / count - mean_x.square()).clamp_min(0.0)
    prior_cfg = config.get("priors", {})
    ridge = float(prior_cfg.get("atlas_ridge", 1e-3))
    slope_clip = float(prior_cfg.get("atlas_slope_clip", 3.0))
    slope = (covariance / (variance + ridge)).clamp(-slope_clip, slope_clip)
    bias = mean_y - slope * mean_x
    payload = {
        "slope": slope.float(),
        "bias": bias.float(),
        "sample_ids": sample_ids,
        "image_size": list(image_size),
        "source_split": "train",
        "num_samples": count,
        "ridge": ridge,
        "slope_clip": slope_clip,
        "preprocess_signature": signature,
    }
    torch.save(payload, path)
    return payload


def save_residual_prior(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _expected_residual_source(config: Dict[str, Any], repo_root: Path) -> Optional[Path]:
    value = config.get("data", {}).get("residual_prior_source_checkpoint")
    return resolve_path(repo_root, value) if value else None


def load_residual_prior(
    config: Dict[str, Any],
    repo_root: Path,
    train_dataset: Any,
    *,
    rebuild: bool = False,
) -> Optional[Dict[str, Any]]:
    if train_dataset is None:
        return None
    path = _prior_path(
        config, repo_root, "residual_prior_path", "data/cache/exp027_voxel_residual_train.pt"
    )
    source_checkpoint = _expected_residual_source(config, repo_root)
    if not path.exists() or rebuild:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signature = _preprocess_signature(config, repo_root, train_dataset)
    if not _compatible(
        payload,
        _dataset_ids(train_dataset),
        config["data"]["image_size"],
        signature,
        source_checkpoint,
    ):
        return None
    return payload


@torch.no_grad()
def fit_residual_prior(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    device: torch.device,
    *,
    sample_ids: Optional[list[str]] = None,
    source_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    image_size = tuple(int(v) for v in config["data"]["image_size"])
    patch_batch_size = int(config["training"]["patch_batch_size"])
    sum_d = torch.zeros((1, 1, *image_size), dtype=torch.float64)
    sum_r = torch.zeros_like(sum_d)
    sum_dd = torch.zeros_like(sum_d)
    sum_dr = torch.zeros_like(sum_d)
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True).float()
        table = batch["table"].to(device, non_blocking=True).float()
        table_mask = batch["mask"].to(device, non_blocking=True).float()
        target = batch["label"].to(device, non_blocking=True).float()
        prediction, auxiliary = model.predict_direct(
            image,
            table,
            table_mask,
            patch_batch_size,
            use_table=True,
            use_global=True,
            use_local_refiner=True,
            use_image_refiner=True,
            clamp_output=False,
        )
        delta = (auxiliary["atlas_pet"] - prediction).float()
        residual = (target - prediction).float()
        sum_d += delta.double().sum(dim=0, keepdim=True)
        sum_r += residual.double().sum(dim=0, keepdim=True)
        sum_dd += delta.double().square().sum(dim=0, keepdim=True)
        sum_dr += (delta.double() * residual.double()).sum(dim=0, keepdim=True)
        count += int(image.shape[0])
    if count == 0:
        raise ValueError("Cannot fit Exp027 residual prior from an empty loader")

    mean_d = sum_d / count
    mean_r = sum_r / count
    covariance = sum_dr / count - mean_d * mean_r
    variance = (sum_dd / count - mean_d.square()).clamp_min(0.0)
    prior_cfg = config.get("priors", {})
    ridge = float(prior_cfg.get("residual_ridge", 1e-3))
    beta_min = float(prior_cfg.get("residual_beta_min", -0.10))
    beta_max = float(prior_cfg.get("residual_beta_max", 0.30))
    beta = (covariance / (variance + count * ridge)).clamp(beta_min, beta_max)
    bias = mean_r - beta * mean_d
    source_path = Path(source_checkpoint).resolve() if source_checkpoint else None
    return {
        "atlas_residual_beta": beta.float(),
        "atlas_residual_bias": bias.float(),
        "sample_ids": sample_ids or [],
        "image_size": list(image_size),
        "source_split": "train",
        "num_samples": count,
        "ridge": ridge,
        "beta_clip": [beta_min, beta_max],
        "source_checkpoint": str(source_path) if source_path else None,
        "source_checkpoint_fingerprint": (
            _checkpoint_fingerprint(source_path) if source_path is not None and source_path.exists() else None
        ),
        "preprocess_signature": _preprocess_signature(
            config, Path(config["experiment"].get("repo_root", Path.cwd())), getattr(loader, "dataset", None)
        ) if getattr(loader, "dataset", None) is not None else None,
    }
