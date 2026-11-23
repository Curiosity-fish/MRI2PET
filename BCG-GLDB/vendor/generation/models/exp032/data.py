from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.paired_fdg_pet_dataset import PairedFDGPETDataset


def resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_labelmap(path: Path, target_shape: tuple[int, int, int]) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Parc98 labelmap not found: {path}")
    values = nib.load(str(path)).get_fdata(dtype="float32")
    labels = torch.from_numpy(values).long()[None, None]
    if tuple(labels.shape[-3:]) != tuple(target_shape):
        labels = F.interpolate(labels.float(), size=target_shape, mode="nearest").long()
    labels = labels.squeeze(0).contiguous()
    unique = set(int(v) for v in torch.unique(labels).tolist())
    if not unique.issubset(set(range(99))):
        raise ValueError(f"Parc98 labelmap contains labels outside 0..98: {sorted(unique)}")
    return labels


class Exp032Dataset(Dataset):
    def __init__(self, base: PairedFDGPETDataset, parc98: torch.Tensor, reconstruction_key: str = "pet") -> None:
        self.base = base
        self.parc98 = parc98.float().contiguous()
        self.reconstruction_key = str(reconstruction_key)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        image = sample["input"] if self.reconstruction_key == "mri" else sample["target"]
        brain_mask = (image.abs() > 1e-6).float()
        return {
            self.reconstruction_key: image,
            "brain_mask": brain_mask,
            "parc98": self.parc98,
            "subject_id": sample["subject_id"],
            "sample_id": sample["sample_id"],
        }


class CachedExp032Dataset(Dataset):
    """Exp032Dataset variant that loads preprocessed volumes from an exp033-style cache.

    Cache layout: {cache_dir}/{split}/{sample_id}__{spatial}.pt with tensor keys
    input/target/brain_mask (float32 volumes). Falls back to the base dataset when
    a cache file is missing (e.g. the small val_fold* subsets).
    """

    def __init__(self, base: PairedFDGPETDataset, parc98: torch.Tensor, cache_dir: str | Path, reconstruction_key: str = "pet") -> None:
        self.base = base
        self.parc98 = parc98.float().contiguous()
        self.cache_dir = Path(cache_dir)
        self.spatial = "x".join(str(v) for v in base.spatial_size)
        self.reconstruction_key = str(reconstruction_key)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.base.df.iloc[index]
        cache_path = self.cache_dir / self.base.split / f"{row['sample_id']}__{self.spatial}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            image = payload["input" if self.reconstruction_key == "mri" else "target"].float()
            brain_mask = payload.get("brain_mask")
            if brain_mask is None:
                brain_mask = (image.abs() > 1e-6).float()
            else:
                brain_mask = brain_mask.float()
        else:
            sample = self.base[index]
            image = (sample["input"] if self.reconstruction_key == "mri" else sample["target"]).float()
            brain_mask = (image.abs() > 1e-6).float()
        return {
            self.reconstruction_key: image,
            "brain_mask": brain_mask,
            "parc98": self.parc98,
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
        }


def build_dataset(
    config: dict,
    repo_root: Path,
    split: str,
) -> Exp032Dataset:
    data = config["data"]
    target_shape = tuple(int(v) for v in data["image_size"])
    base = PairedFDGPETDataset(
        pairs_csv=resolve_path(repo_root, data["pairs_csv"]),
        split_csv=resolve_path(repo_root, data["split_csv"]),
        split=split,
        fold=data.get("fold"),
        spatial_size=target_shape,
        max_samples=data.get(f"{split}_max_samples"),
        normalize_targets=bool(data.get("normalize_targets", False)),
    )
    labelmap = load_labelmap(resolve_path(repo_root, data["parc98_labelmap_path"]), target_shape)
    cache_dir = data.get("cache_dir")
    reconstruction_key = str(data.get("reconstruction_key", "pet"))
    if reconstruction_key not in {"pet", "mri"}:
        raise ValueError(f"Unsupported reconstruction_key={reconstruction_key}")
    if cache_dir:
        return CachedExp032Dataset(base, labelmap, cache_dir, reconstruction_key)
    return Exp032Dataset(base, labelmap, reconstruction_key)
