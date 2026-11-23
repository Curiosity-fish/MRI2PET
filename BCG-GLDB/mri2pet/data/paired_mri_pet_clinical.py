from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mri2pet.data.paired_mri_pet import _build_brain_mask, _load_volume, _resize_volume


def prepare_table_dataframe(
    table_csv: str | Path,
    table_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(table_csv).copy()
    if "VISDATE" not in df.columns and "VISIT_DATE" in df.columns:
        df = df.rename(columns={"VISIT_DATE": "VISDATE"})
    if "PTID" not in df.columns:
        raise ValueError("table_csv must contain PTID.")

    # Some harmonized tables provide one static clinical row per subject rather
    # than visit-level records.  Keep the existing PTID+VISDATE behavior when
    # dates exist, otherwise expose one deterministic static row per PTID.
    if "VISDATE" not in df.columns:
        if table_cols is None:
            exclude = {"RID", "PTID", "SUBJECT_ID"}
            table_cols = [
                c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not table_cols:
                raise ValueError("No numeric table columns found in table_csv.")
        else:
            missing = [c for c in table_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing table columns: {missing}")
        grouped = df[["PTID"] + table_cols].copy()
        grouped[table_cols] = grouped.groupby("PTID", sort=False)[table_cols].transform(
            lambda group: group.bfill().ffill()
        )
        grouped = grouped.drop_duplicates("PTID", keep="first").reset_index(drop=True)
        grouped["DATE_KEY"] = "__static__"
        return grouped[["PTID", "DATE_KEY"] + table_cols], table_cols

    dt = pd.to_datetime(df["VISDATE"], errors="coerce")
    df = df.loc[~dt.isna()].copy()
    df["DATE_KEY"] = dt.loc[~dt.isna()].dt.strftime("%Y-%m-%d")

    if table_cols is None:
        exclude = {"RID", "PTID", "VISCODE2", "VISDATE", "DATE_KEY"}
        table_cols = [
            c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not table_cols:
            raise ValueError("No numeric table columns found in table_csv.")
    else:
        missing = [c for c in table_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing table columns: {missing}")

    group_cols = ["PTID", "DATE_KEY"]
    df = df[group_cols + table_cols].copy()
    df[table_cols] = df.groupby(group_cols, sort=False)[table_cols].transform(
        lambda group: group.bfill().ffill()
    )
    grouped = df.drop_duplicates(group_cols, keep="first").reset_index(drop=True)
    return grouped, table_cols


def build_tabular_stats(
    table_df: pd.DataFrame,
    table_cols: list[str],
    standardize: bool = True,
) -> dict:
    x = table_df[table_cols].astype(np.float32).to_numpy(copy=True)
    medians = np.nanmedian(x, axis=0).astype(np.float32)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)
    x_imp = np.where(np.isnan(x), medians[None, :], x).astype(np.float32)
    stats = {"columns": list(table_cols), "medians": medians}
    if standardize:
        means = x_imp.mean(axis=0).astype(np.float32)
        stds = x_imp.std(axis=0).astype(np.float32)
        stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
        stats["means"] = means
        stats["stds"] = stds
    return stats


def select_table_rows_for_samples(
    table_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    max_delta_days: Optional[int],
) -> pd.DataFrame:
    """Select only clinical rows eligible for the supplied image samples.

    This helper is used when fitting imputation and standardization statistics.
    In particular, a training dataset must not fit those statistics on rows
    belonging to validation or final-validation subjects.
    """

    subjects = set(sample_df["subject_id"].astype(str))
    candidate_df = table_df[table_df["PTID"].astype(str).isin(subjects)].copy()
    if candidate_df.empty:
        raise ValueError("No clinical rows match the subjects in the requested split.")

    if max_delta_days is None or bool((candidate_df["DATE_KEY"] == "__static__").all()):
        return candidate_df

    by_subject = {
        subject: group
        for subject, group in candidate_df.groupby(candidate_df["PTID"].astype(str), sort=False)
    }
    selected_indices: set[int] = set()
    for _, sample in sample_df.iterrows():
        subject = str(sample["subject_id"])
        candidates = by_subject.get(subject)
        sample_date = pd.to_datetime(sample.get("t1_date", ""), errors="coerce")
        if candidates is None or pd.isna(sample_date):
            continue
        dates = pd.to_datetime(candidates["DATE_KEY"], errors="coerce")
        valid = ~dates.isna()
        if not bool(valid.any()):
            continue
        deltas = (dates.loc[valid] - sample_date).abs().dt.days
        best_index = deltas.idxmin()
        if int(deltas.loc[best_index]) <= int(max_delta_days):
            selected_indices.add(int(best_index))

    if not selected_indices:
        raise ValueError(
            "No clinical rows fall within max_table_delta_days for the requested split."
        )
    return candidate_df.loc[sorted(selected_indices)].copy()


class PairedT1FDGTableDataset(Dataset):
    def __init__(
        self,
        pairs_csv: str | Path,
        split_csv: str | Path,
        table_csv: str | Path,
        split: str,
        fold: Optional[int],
        spatial_size: Tuple[int, int, int],
        tabular_stats: Optional[dict] = None,
        table_cols: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
        include_sample_ids: Optional[list[str]] = None,
        standardize_table: bool = True,
        add_missing_mask: bool = True,
        cache_dir: Optional[str | Path] = None,
        max_table_delta_days: Optional[int] = None,
    ) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.split_csv = Path(split_csv)
        self.table_csv = Path(table_csv)
        self.split = str(split)
        self.fold = fold
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.standardize_table = bool(standardize_table)
        self.add_missing_mask = bool(add_missing_mask)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_table_delta_days = (
            None if max_table_delta_days is None else int(max_table_delta_days)
        )

        pairs_df = pd.read_csv(self.pairs_csv)
        split_df = pd.read_csv(self.split_csv)
        df = pairs_df.merge(
            split_df, on=["subject_id", "sample_id"], how="inner", validate="one_to_one"
        )

        available_splits = set(df["split"].astype(str).unique().tolist())
        has_explicit_val = "val" in available_splits
        if split == "train":
            df = df[df["split"] == "train"].copy()
            # Explicit train/val/final_val files do not use fold-based
            # exclusion.  Preserve the old behavior for legacy files.
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
        if include_sample_ids:
            include = {str(v) for v in include_sample_ids}
            df = df[df["sample_id"].astype(str).isin(include)].copy()
        df = df.sort_values(["subject_id", "sample_id"]).reset_index(drop=True)
        if max_samples is not None:
            df = df.head(int(max_samples)).copy()
        if df.empty:
            raise ValueError(f"No samples found for split={split}, fold={fold}")
        self.df = df

        table_df, resolved_cols = prepare_table_dataframe(self.table_csv, table_cols=table_cols)
        self.table_cols = resolved_cols
        if tabular_stats is None:
            stats_df = select_table_rows_for_samples(
                table_df,
                self.df,
                self.max_table_delta_days,
            )
            self.tabular_stats = build_tabular_stats(
                stats_df,
                self.table_cols,
                standardize=self.standardize_table,
            )
        else:
            self.tabular_stats = tabular_stats

        self.table_index: Dict[tuple[str, str], pd.Series] = {}
        self.table_by_ptid: Dict[str, list[pd.Series]] = {}
        for _, row in table_df.iterrows():
            self.table_index[(str(row["PTID"]), str(row["DATE_KEY"]))] = row
            self.table_by_ptid.setdefault(str(row["PTID"]), []).append(row)

    def _cache_key(self, row: pd.Series) -> str:
        spatial = "x".join(str(v) for v in self.spatial_size)
        return f"{row['sample_id']}__{spatial}.pt"

    def _cache_path(self, row: pd.Series) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.split / self._cache_key(row)

    def _load_or_build_image_sample(self, row: pd.Series) -> Dict[str, torch.Tensor]:
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
        sample = {"input": t1, "target": fdg, "brain_mask": brain_mask}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, cache_path)
        return sample

    def _get_table_tensors(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        ptid = str(row["subject_id"])
        date_key = str(row.get("table_date_key", ""))
        record = None
        if self.max_table_delta_days is not None:
            sample_date = pd.to_datetime(row.get("t1_date", ""), errors="coerce")
            candidates = self.table_by_ptid.get(ptid, [])
            static = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate["DATE_KEY"]) == "__static__"
                ),
                None,
            )
            if static is not None:
                record = static
            elif not pd.isna(sample_date) and candidates:
                dated = [
                    (candidate, pd.to_datetime(candidate["DATE_KEY"], errors="coerce"))
                    for candidate in candidates
                ]
                dated = [(candidate, date) for candidate, date in dated if not pd.isna(date)]
                if dated:
                    best, date = min(dated, key=lambda item: abs((item[1] - sample_date).days))
                    if abs((date - sample_date).days) <= self.max_table_delta_days:
                        record = best
        else:
            record = self.table_index.get(
                (ptid, date_key), self.table_index.get((ptid, "__static__"), None)
            )

        f = len(self.table_cols)
        if record is None:
            values = np.zeros((f,), dtype=np.float32)
            miss = np.ones((f,), dtype=np.float32)
        else:
            values = record[self.table_cols].astype(np.float32).to_numpy(copy=True)
            miss = np.isnan(values).astype(np.float32)
            medians = self.tabular_stats["medians"]
            values = np.where(np.isnan(values), medians, values).astype(np.float32)
            if (
                self.standardize_table
                and "means" in self.tabular_stats
                and "stds" in self.tabular_stats
            ):
                values = (values - self.tabular_stats["means"]) / self.tabular_stats["stds"]

        return torch.from_numpy(values), torch.from_numpy(miss)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        image_sample = self._load_or_build_image_sample(row)
        table, table_mask = self._get_table_tensors(row)
        return {
            "input": image_sample["input"],
            "target": image_sample["target"],
            "brain_mask": image_sample["brain_mask"],
            "table": table.float(),
            "table_mask": (
                table_mask.float() if self.add_missing_mask else torch.zeros_like(table).float()
            ),
            "subject_id": row["subject_id"],
            "sample_id": row["sample_id"],
        }
