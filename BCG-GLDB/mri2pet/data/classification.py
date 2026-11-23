"""Subject-level five-fold data access with train-fold-only clinical statistics."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

TABLE_COLS = [
    "PTDEMOG_PTGENDER",
    "PTDEMOG_PTEDUCAT",
    "AGE_derived",
    "APOE4_COUNT",
    "APOE4_CARRIER",
    "FAMILY_HISTORY_AD",
    "PARENT_HISTORY_AD",
    "SIBLING_HISTORY_AD",
    "MMSE_MMSCORE",
    "ADAS_TOTAL13",
    "MOCA_MOCA",
    "MEDHIST_MH2NEURL",
    "MEDHIST_MH3HEAD",
    "MEDHIST_MH4CARD",
    "MEDHIST_MH9ENDO",
    "MEDHIST_MH14ALCH",
    "MEDHIST_MH16SMOK",
    "VITALS_VSWEIGHT",
    "VITALS_VSHEIGHT",
    "BMI_derived",
    "VITALS_VSBPSYS",
    "VITALS_VSBPDIA",
    "VITALS_VSPULSE",
    "PHYSICAL_PXABNORM",
    "NEUROEXM_NXABNORM",
]


def load_index(
    labels_csv,
    split_csv,
    *,
    train_splits=("train", "val"),
    test_split="final_val",
    expected_train=899,
    expected_test=96,
):
    labels = pd.read_csv(labels_csv)
    split = pd.read_csv(split_csv)
    frame = labels.merge(split, on=["subject_id", "sample_id"], validate="one_to_one")
    frame = frame[frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1, 2}):
        raise ValueError("CN/MCI/AD labels must be encoded as 0/1/2")
    training = (
        frame[frame["split"].isin(train_splits)]
        .sort_values(["subject_id", "sample_id"])
        .reset_index(drop=True)
    )
    testing = (
        frame[frame["split"] == test_split]
        .sort_values(["subject_id", "sample_id"])
        .reset_index(drop=True)
    )
    if expected_train is not None and len(training) != expected_train:
        raise ValueError(f"Expected {expected_train} training-pool samples, found {len(training)}")
    if expected_test is not None and len(testing) != expected_test:
        raise ValueError(f"Expected {expected_test} fixed-test samples, found {len(testing)}")
    if set(training.subject_id.astype(str)) & set(testing.subject_id.astype(str)):
        raise ValueError("Training-pool and fixed-test subjects overlap")
    return training, testing


def read_clinical_table(table_csv):
    table = pd.read_csv(table_csv).copy()
    table.columns = [str(column).lstrip("\ufeff") for column in table.columns]
    missing = set(["PTID", *TABLE_COLS]) - set(table.columns)
    if missing:
        raise ValueError(f"Clinical table lacks fields: {sorted(missing)}")
    if "VISDATE" in table.columns:
        table["DATE_KEY"] = pd.to_datetime(table["VISDATE"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    else:
        table["DATE_KEY"] = "__static__"
    table["PTID"] = table["PTID"].astype(str)
    table[TABLE_COLS] = table[TABLE_COLS].apply(pd.to_numeric, errors="coerce")
    return table[["PTID", "DATE_KEY", *TABLE_COLS]].drop_duplicates(["PTID", "DATE_KEY"])


def clinical_key(sample, table):
    subject = str(sample["subject_id"])
    date = str(sample.get("table_date_key", "__static__"))
    if pd.isna(sample.get("table_date_key", None)):
        date = "__static__"
    selected = table[(table.PTID == subject) & (table.DATE_KEY == date)]
    if selected.empty and date != "__static__":
        selected = table[(table.PTID == subject) & (table.DATE_KEY == "__static__")]
    return None if selected.empty else selected.iloc[0]


def fit_clinical_stats(train_frame, table):
    """Only the current fold's training samples contribute to fitted statistics."""
    rows = []
    for _, sample in train_frame.iterrows():
        matched = clinical_key(sample, table)
        if matched is not None:
            rows.append(matched[TABLE_COLS].to_numpy(np.float32))
    if not rows:
        raise ValueError("No training-fold clinical rows match the image samples")
    values = np.asarray(rows, dtype=np.float32)
    median = np.nanmedian(values, axis=0)
    median = np.nan_to_num(median, nan=0.0)
    filled = np.where(np.isnan(values), median[None], values)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std[std < 1e-6] = 1.0
    return {
        "median": median.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "fit_sample_ids": train_frame.sample_id.astype(str).tolist(),
    }


def load_volume(path, shape, root):
    value = Path(str(path)).expanduser()
    if not value.is_absolute():
        value = root / value
    if not value.exists():
        raise FileNotFoundError(value)
    if value.suffix == ".pt":
        payload = torch.load(value, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payload = payload.get("prediction", payload.get("pet", payload.get("image", payload)))
        image = torch.as_tensor(payload).float()
    else:
        image = torch.from_numpy(np.asarray(nib.load(str(value)).dataobj, dtype=np.float32).copy())
    while image.ndim > 3:
        if image.shape[0] != 1:
            raise ValueError(f"Unexpected extra image dimension in {value}: {tuple(image.shape)}")
        image = image.squeeze(0)
    if image.ndim != 3:
        raise ValueError(f"Expected one volume in {value}")
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
    image = F.interpolate(image[None, None], size=shape, mode="trilinear", align_corners=False)[0]
    return image


class ClassificationDataset(Dataset):
    def __init__(
        self,
        frame,
        clinical_table,
        stats,
        root,
        *,
        modality="mri_pet",
        pet_path_col="fdg_pet_path",
        spatial=(160, 192, 160),
        cache_dir=None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.clinical_table = clinical_table
        self.stats = stats
        self.root = Path(root)
        self.modality = modality
        self.pet_path_col = pet_path_col
        self.spatial = tuple(spatial)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.modality not in {"mri", "pet", "mri_pet"}:
            raise ValueError("modality must be mri, pet or mri_pet")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        sample = self.frame.iloc[index]
        image = None
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{sample.sample_id}.pt"
            if cached.exists():
                payload = torch.load(cached, map_location="cpu", weights_only=True)
                key = {"mri": "mri", "pet": "pet", "mri_pet": "mri_pet"}[self.modality]
                image = payload.get(key)
                if image is not None and tuple(image.shape[-3:]) != self.spatial:
                    raise ValueError(
                        f"Stale cache has wrong dimensions: {cached} {tuple(image.shape)}"
                    )
        if image is None:
            volumes = []
            if self.modality in {"mri", "mri_pet"}:
                volumes.append(load_volume(sample.t1_path, self.spatial, self.root))
            if self.modality in {"pet", "mri_pet"}:
                volumes.append(load_volume(sample[self.pet_path_col], self.spatial, self.root))
            image = torch.cat(volumes, dim=0)
        matched = clinical_key(sample, self.clinical_table)
        if matched is None:
            values = np.full(25, np.nan, dtype=np.float32)
        else:
            values = matched[TABLE_COLS].to_numpy(np.float32)
        missing = np.isnan(values).astype(np.float32)
        values = np.where(np.isnan(values), self.stats["median"], values)
        values = ((values - self.stats["mean"]) / self.stats["std"]).astype(np.float32)
        return {
            "image": image.float(),
            "table": torch.from_numpy(values),
            "table_mask": torch.from_numpy(missing),
            "label": int(sample.label),
            "sample_id": str(sample.sample_id),
            "subject_id": str(sample.subject_id),
        }
