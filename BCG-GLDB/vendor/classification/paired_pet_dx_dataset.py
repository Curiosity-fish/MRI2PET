from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .paired_t1_dx_dataset import _load_volume, _resize_volume


class PairedPETDiagnosisDataset(Dataset):
    def __init__(
        self,
        labels_csv: str | Path,
        split_csv: str | Path,
        split: str,
        fold: Optional[int],
        spatial_size: Tuple[int, int, int],
        max_samples: Optional[int] = None,
        include_sample_ids: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
        cache_dir: Optional[str | Path] = './data/classier/data/cache/origin',
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.split_csv = Path(split_csv)
        self.split = str(split)
        self.fold = fold
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.class_names = class_names or ["CN", "MCI", "AD"]
        self.cache_dir = Path(cache_dir) if cache_dir else None

        labels_df = pd.read_csv(self.labels_csv)
        split_df = pd.read_csv(self.split_csv)
        df = labels_df.merge(split_df, on=["subject_id", "sample_id"], how="inner", validate="one_to_one")

        if split == "train":
            df = df[df["split"] == "train"].copy()
            if fold is not None:
                df = df[df["fold"] != int(fold)].copy()
        elif split == "val":
            if fold is None:
                df = df[df["split"] == "val"].copy()
            else:
                df = df[(df["split"] == "train") & (df["fold"] == int(fold))].copy()
        elif split == "final_val":
            df = df[df["split"] == "final_val"].copy()
        else:
            raise ValueError(f"Unsupported split: {split}")

        if include_sample_ids:
            include_set = {str(v) for v in include_sample_ids}
            df = df[df["sample_id"].astype(str).isin(include_set)].copy()

        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        df = df[df["fdg_pet_path"].notna()].copy()
        df = df.sort_values(["subject_id", "sample_id"]).reset_index(drop=True)

        if max_samples is not None:
            df = df.head(int(max_samples)).copy()

        if df.empty:
            raise ValueError(f"No labeled PET samples found for split={split}, fold={fold}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        cp = self.cache_dir / f"{row['sample_id']}.pt" if self.cache_dir else None
        if cp and cp.exists(): pet = torch.load(cp, map_location='cpu', weights_only=True)['pet'].float()
        else:
            path = str(row["fdg_pet_path"])
            if path.endswith('.pt'):
                obj = torch.load(path, map_location='cpu', weights_only=False)
                pet = obj.get('prediction', obj.get('pet', obj))
                while pet.ndim > 4:
                    pet = pet.squeeze(0)
                if pet.ndim == 4: pet = pet.float()
                elif pet.ndim == 3: pet = pet.unsqueeze(0).float()
                else: raise ValueError(f'Unexpected generated PET shape {tuple(pet.shape)}')
                pet = torch.nan_to_num(pet.float(), nan=0.0, posinf=1.0, neginf=-1.0)
                pet = _resize_volume(pet, self.spatial_size)
            else: pet = _resize_volume(_load_volume(path), self.spatial_size)
        label = int(row["label"])
        label_name = row.get("label_name", self.class_names[label] if 0 <= label < len(self.class_names) else str(label))
        return {
            "input": pet,
            "label": torch.tensor(label, dtype=torch.long),
            "label_name": label_name,
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
            "fdg_date": row.get("fdg_date", ""),
            "dx_date": row.get("dx_date", ""),
            "dx_delta_days": row.get("dx_delta_days", ""),
        }
