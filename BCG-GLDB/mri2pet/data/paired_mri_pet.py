from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load_volume(path: str) -> torch.Tensor:
    array = nib.load(path).get_fdata(dtype="float32")
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape={array.shape} for {path}")
    tensor = torch.from_numpy(array).unsqueeze(0).float()
    return tensor


def _resize_volume(volume: torch.Tensor, spatial_size: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(volume.shape[-3:]) == tuple(spatial_size):
        return volume
    return F.interpolate(
        volume.unsqueeze(0),
        size=spatial_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def _normalize_volume(volume: torch.Tensor) -> torch.Tensor:
    mean = volume.mean()
    std = volume.std()
    if torch.isnan(std) or std < 1e-6:
        return volume - mean
    return (volume - mean) / std


def _build_brain_mask(t1: torch.Tensor, fdg: torch.Tensor) -> torch.Tensor:
    return ((t1.abs() > 1e-6) | (fdg.abs() > 1e-6)).float()


class PairedT1FDGDataset(Dataset):
    def __init__(
        self,
        pairs_csv: str | Path,
        split_csv: str | Path,
        split: str,
        fold: Optional[int],
        spatial_size: Tuple[int, int, int],
        max_samples: Optional[int] = None,
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        include_sample_ids: Optional[list[str]] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.split_csv = Path(split_csv)
        self.split = split
        self.fold = fold
        self.spatial_size = tuple(int(v) for v in spatial_size)
        # Keep these flags for backward compatibility with older configs, but
        # the current T1-FDG pipeline no longer applies per-sample z-score.
        self.normalize_inputs = normalize_inputs
        self.normalize_targets = normalize_targets
        self.cache_dir = Path(cache_dir) if cache_dir else None

        pairs_df = pd.read_csv(self.pairs_csv)
        split_df = pd.read_csv(self.split_csv)
        df = pairs_df.merge(
            split_df, on=["subject_id", "sample_id"], how="inner", validate="one_to_one"
        )

        available_splits = set(df["split"].astype(str).unique().tolist())
        has_explicit_val = "val" in available_splits
        if split == "train":
            df = df[df["split"] == "train"].copy()
            # New thesis split files contain an explicit val subset.  Keep the
            # legacy fold exclusion only for old files that do not contain it.
            if fold is not None and not has_explicit_val:
                df = df[df["fold"] != int(fold)].copy()
        elif split == "val":
            if has_explicit_val:
                df = df[df["split"] == "val"].copy()
            else:
                if fold is None:
                    raise ValueError("Validation split requires a fold for legacy split files.")
                df = df[(df["split"] == "train") & (df["fold"] == int(fold))].copy()
        elif str(split).startswith("val_fold"):
            requested_fold = int(str(split).replace("val_fold", ""))
            df = df[df["split"].astype(str) == str(split)].copy()
            if df.empty:
                df = df[(df["split"] == "train") & (df["fold"] == requested_fold)].copy()
        elif split == "final_val":
            df = df[df["split"] == "final_val"].copy()
        elif split in available_splits:
            df = df[df["split"] == str(split)].copy()
        else:
            raise ValueError(f"Unsupported split: {split}")

        df = df[df["has_fdg_pet"] == True].copy()
        if include_sample_ids:
            include_set = {str(v) for v in include_sample_ids}
            df = df[df["sample_id"].astype(str).isin(include_set)].copy()
        df = df.sort_values(["subject_id", "sample_id"]).reset_index(drop=True)

        if max_samples is not None:
            df = df.head(int(max_samples)).copy()

        if df.empty:
            raise ValueError(f"No samples found for split={split}, fold={fold}")

        self.df = df

    def _cache_key(self, row: pd.Series) -> str:
        spatial = "x".join(str(v) for v in self.spatial_size)
        return f"{row['sample_id']}__{spatial}.pt"

    def _cache_path(self, row: pd.Series) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.split / self._cache_key(row)

    def _load_or_build_sample(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        cache_path = self._cache_path(row)
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            return {
                "input": payload["input"].float(),
                "target": payload["target"].float(),
                "brain_mask": payload["brain_mask"].float(),
            }

        t1 = _resize_volume(_load_volume(row["t1_path"]), self.spatial_size)
        fdg = _resize_volume(_load_volume(row["fdg_pet_path"]), self.spatial_size)
        brain_mask = _build_brain_mask(t1, fdg)
        sample = {
            "input": t1,
            "target": fdg,
            "brain_mask": brain_mask,
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, cache_path)
        return sample

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        sample = self._load_or_build_sample(row)

        return {
            "input": sample["input"],
            "target": sample["target"],
            "brain_mask": sample["brain_mask"],
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
        }
