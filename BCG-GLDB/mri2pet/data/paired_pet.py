from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch

from mri2pet.data.paired_mri_pet import _load_volume, _normalize_volume, _resize_volume


class PairedFDGPETDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs_csv: str | Path,
        split_csv: str | Path,
        split: str,
        fold: Optional[int],
        spatial_size: Tuple[int, int, int],
        max_samples: Optional[int] = None,
        normalize_targets: bool = False,
    ) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.split_csv = Path(split_csv)
        self.split = split
        self.fold = fold
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.normalize_targets = bool(normalize_targets)

        pairs_df = pd.read_csv(self.pairs_csv)
        split_df = pd.read_csv(self.split_csv)
        df = pairs_df.merge(
            split_df, on=["subject_id", "sample_id"], how="inner", validate="one_to_one"
        )

        available_splits = set(df["split"].astype(str).unique().tolist())
        has_explicit_val = "val" in available_splits
        if split == "train":
            df = df[df["split"] == "train"].copy()
            if fold is not None and not has_explicit_val:
                df = df[df["fold"] != int(fold)].copy()
        elif split == "val":
            if has_explicit_val:
                df = df[df["split"] == "val"].copy()
            else:
                if fold is None:
                    raise ValueError("Validation split requires a fold for legacy split files.")
                df = df[(df["split"] == "train") & (df["fold"] == int(fold))].copy()
        elif split == "final_val":
            df = df[df["split"] == "final_val"].copy()
        elif split in available_splits:
            df = df[df["split"] == str(split)].copy()
        else:
            raise ValueError(f"Unsupported split: {split}")

        df = df[df["has_fdg_pet"] == True].copy()
        df = df.sort_values(["subject_id", "sample_id"]).reset_index(drop=True)

        if max_samples is not None:
            df = df.head(int(max_samples)).copy()

        if df.empty:
            raise ValueError(f"No FDG-PET samples found for split={split}, fold={fold}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        pet = _resize_volume(_load_volume(row["fdg_pet_path"]), self.spatial_size)
        if self.normalize_targets:
            pet = _normalize_volume(pet)
        return {
            "target": pet,
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
            "table_date_key": row.get("table_date_key"),
            "has_table": bool(row.get("has_table", False)),
        }
