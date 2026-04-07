from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    CropForegroundd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
)

# 兼容某些 Windows/Intel MKL 环境的报错
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")


# =========================
#   Tabular 统计信息
# =========================
@dataclass
class TabularStats:
    """
    用于表格特征的填充与标准化参数。
    - medians: 用于缺失值填充（列中位数）
    - means/stds: 用于 z-score 标准化（可选）
    """
    columns: List[str]
    medians: np.ndarray
    means: Optional[np.ndarray] = None
    stds: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        d = {
            "columns": self.columns,
            "medians": self.medians.tolist(),
        }
        if self.means is not None:
            d["means"] = self.means.tolist()
        if self.stds is not None:
            d["stds"] = self.stds.tolist()
        return d

    @staticmethod
    def from_dict(d: Dict) -> "TabularStats":
        return TabularStats(
            columns=list(d["columns"]),
            medians=np.asarray(d["medians"], dtype=np.float32),
            means=np.asarray(d["means"], dtype=np.float32) if "means" in d else None,
            stds=np.asarray(d["stds"], dtype=np.float32) if "stds" in d else None,
        )


def _parse_folder_key(folder_name: str) -> Optional[Tuple[str, str]]:
    """
    从样本文件夹名解析 (PTID, date_key)：
      - PTID: 002_S_1268
      - date_key: 2011-04-13

    返回 None 表示不符合规则。
    """
    m = re.match(r"(\d+_S_\d+)-([A-Za-z0-9]+)-(\d{4}[_\-]\d{2}[_\-]\d{2})$", folder_name)
    if not m:
        return None
    ptid = m.group(1)
    date_raw = m.group(3).replace("_", "-")
    dt = pd.to_datetime(date_raw, errors="coerce")
    if pd.isna(dt):
        return None
    return ptid, dt.strftime("%Y-%m-%d")


def _check_single_channel_nii(file_path: Union[str, Path]) -> bool:
    """检查 NIfTI 是否为单通道（3D 或 4D 且 channel=1）。"""
    try:
        img = nib.load(str(file_path))
        shp = img.shape
        if len(shp) == 3:
            return True
        if len(shp) == 4 and shp[-1] == 1:
            return True
        return False
    except Exception:
        return False


def _prepare_table(
    table_csv: Union[str, Path],
    table_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取 table.csv，生成用于匹配的 key：
    - PTID
    - DATE_KEY: YYYY-MM-DD（由 VISDATE 解析）
    同时返回最终使用的表格列 columns（数值特征列）。
    """
    df = pd.read_csv(table_csv)

    if "PTID" not in df.columns or "VISDATE" not in df.columns:
        raise ValueError(f"table_csv 必须包含 PTID 与 VISDATE 字段，当前字段为: {list(df.columns)}")

    # 解析日期
    dt = pd.to_datetime(df["VISDATE"], errors="coerce")
    df = df.loc[~dt.isna()].copy()
    df["DATE_KEY"] = dt.loc[~dt.isna()].dt.strftime("%Y-%m-%d")

    # 选择特征列
    if table_cols is None:
        exclude = {"RID", "PTID", "VISDATE", "DATE_KEY"}
        numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            raise ValueError("未在 table_csv 中发现可用的数值特征列。请显式传入 table_cols。")
        table_cols = numeric_cols
    else:
        missing = [c for c in table_cols if c not in df.columns]
        if missing:
            raise ValueError(f"table_cols 中包含 table_csv 不存在的列: {missing}")

    # 对同一 (PTID, DATE_KEY) 可能存在多行：按列取“第一个非空值”，否则 NaN
    def first_non_nan(s: pd.Series):
        non_nan = s.dropna()
        return non_nan.iloc[0] if len(non_nan) else np.nan

    agg = {c: first_non_nan for c in table_cols}
    df_g = (
        df.groupby(["PTID", "DATE_KEY"], as_index=False)
          .agg(agg)
    )

    return df_g, table_cols


def _build_tabular_stats(
    df_g: pd.DataFrame,
    table_cols: List[str],
    standardize: bool = True,
) -> TabularStats:
    """
    计算用于缺失填充与（可选）标准化的统计量。
    注意：这里基于 df_g（已按 PTID+DATE_KEY 聚合）计算。
    """
    X = df_g[table_cols].astype(np.float32).to_numpy(copy=True)

    medians = np.nanmedian(X, axis=0).astype(np.float32)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)

    X_imp = np.where(np.isnan(X), medians[None, :], X).astype(np.float32)

    means = stds = None
    if standardize:
        means = X_imp.mean(axis=0).astype(np.float32)
        stds = X_imp.std(axis=0).astype(np.float32)
        stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)

    return TabularStats(columns=table_cols, medians=medians, means=means, stds=stds)


def _get_tabular_vector(
    row: pd.Series,
    stats: TabularStats,
    add_missing_mask: bool = True,
    standardize: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    给定一行表格，生成表格特征向量（已填充、可选标准化）
    并可返回缺失掩码（1 表示缺失、0 表示原本非缺失）
    """
    x = row[stats.columns].astype(np.float32).to_numpy(copy=True)
    miss = np.isnan(x).astype(np.float32) if add_missing_mask else None
    x = np.where(np.isnan(x), stats.medians, x).astype(np.float32)

    if standardize and (stats.means is not None) and (stats.stds is not None):
        x = (x - stats.means) / stats.stds

    return x, miss


# =========================
#       Dataset
# =========================
class MRI2PETTableDataset(Dataset):
    """
    返回字段：
      - image: MRI tensor, shape = (1, D, H, W)
      - label: PET tensor, shape = (1, D, H, W)
      - table: 表格特征 tensor, shape = (F,) 或 (F + F_mask,) 若 add_missing_mask=True
      - meta: 字符串信息（folder / PTID / DATE_KEY / TABLE_DATE_KEY / TABLE_DELTA_DAYS）
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        table_csv: Union[str, Path],
        desired_shape: Tuple[int, int, int] = (160, 160, 96),
        table_cols: Optional[List[str]] = None,
        add_missing_mask: bool = True,
        standardize_table: bool = True,
        tabular_stats: Optional[Union[TabularStats, Dict]] = None,
        allow_unmatched_table: bool = False,
        strict_channel_check: bool = True,
        match_policy: Literal["exact", "nearest"] = "nearest",
        max_date_delta_days: int = 30,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.desired_shape = tuple(desired_shape)
        self.add_missing_mask = add_missing_mask
        self.standardize_table = standardize_table
        self.allow_unmatched_table = allow_unmatched_table
        self.strict_channel_check = strict_channel_check
        self.match_policy = match_policy
        self.max_date_delta_days = int(max_date_delta_days)

        # 读取并聚合 table
        self.table_df_g, self.table_cols = _prepare_table(table_csv, table_cols=table_cols)

        # 统计量（填充 + 标准化）
        if tabular_stats is None:
            self.tab_stats = _build_tabular_stats(self.table_df_g, self.table_cols, standardize=standardize_table)
        else:
            self.tab_stats = tabular_stats if isinstance(tabular_stats, TabularStats) else TabularStats.from_dict(tabular_stats)
            if list(self.tab_stats.columns) != list(self.table_cols):
                raise ValueError(
                    "tabular_stats.columns 与 table_cols 不一致。"
                    "建议：训练集先生成 stats，再在验证/测试集复用同一 stats。"
                )

        # 快速索引： exact 匹配
        self._table_index: Dict[Tuple[str, str], pd.Series] = {}
        # ptid -> [(date_key, row), ...] 用于 nearest 匹配
        self._table_by_ptid: Dict[str, List[Tuple[str, pd.Series]]] = {}

        for _, r in self.table_df_g.iterrows():
            ptid = str(r["PTID"])
            dkey = str(r["DATE_KEY"])
            self._table_index[(ptid, dkey)] = r
            self._table_by_ptid.setdefault(ptid, []).append((dkey, r))

        # 对每个 PTID 的日期排序，方便 nearest
        for ptid, lst in self._table_by_ptid.items():
            lst.sort(key=lambda t: pd.to_datetime(t[0], errors="coerce"))

        # 过滤有效样本
        self.valid_samples = self._filter_samples()

        # 图像 transform（与现有 loader 接近，但保证 image/label 同步裁剪）
        self.transform = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"], dtype=np.float32),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            Resized(keys=["image", "label"], spatial_size=self.desired_shape),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=False),
            ToTensord(keys=["image", "label"]),
        ])

    def _match_table(self, ptid: str, date_key: str) -> Tuple[Optional[pd.Series], Optional[str], Optional[int]]:
        """
        返回 (row, matched_date_key, delta_days)。
        - exact: 必须完全同日
        - nearest: 同 PTID 下选最近日期，且 |delta_days| <= max_date_delta_days
        """
        if self.match_policy == "exact":
            row = self._table_index.get((ptid, date_key), None)
            return row, (date_key if row is not None else None), (0 if row is not None else None)

        # nearest
        candidates = self._table_by_ptid.get(ptid, [])
        if not candidates:
            return None, None, None

        target = pd.to_datetime(date_key, errors="coerce")
        if pd.isna(target):
            return None, None, None

        best = None  # (abs_delta, delta, matched_date_key, row)
        for dkey, row in candidates:
            d = pd.to_datetime(dkey, errors="coerce")
            if pd.isna(d):
                continue
            delta = int((d - target).days)
            abs_delta = abs(delta)
            if best is None or abs_delta < best[0]:
                best = (abs_delta, delta, dkey, row)

        if best is None:
            return None, None, None

        abs_delta, delta, dkey, row = best
        if abs_delta > self.max_date_delta_days:
            return None, None, None
        return row, dkey, delta

    def _filter_samples(self) -> List[Dict]:
        valid: List[Dict] = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"data_path 不存在: {self.data_path}")

        for folder in os.listdir(self.data_path):
            folder_path = self.data_path / folder
            if not folder_path.is_dir():
                continue

            key = _parse_folder_key(folder)
            if key is None:
                continue
            ptid, date_key = key

            mri_file = folder_path / "MRI.nii.gz"
            pet_file = folder_path / "PET.nii.gz"
            if not (mri_file.exists() and pet_file.exists()):
                continue

            if self.strict_channel_check:
                if not (_check_single_channel_nii(mri_file) and _check_single_channel_nii(pet_file)):
                    continue

            row, table_date_key, table_delta_days = self._match_table(ptid, date_key)
            has_table = row is not None

            if (not has_table) and (not self.allow_unmatched_table):
                continue

            valid.append({
                "folder": str(folder_path),
                "ptid": ptid,
                "date_key": date_key,
                "mri_path": str(mri_file),
                "pet_path": str(pet_file),
                "has_table": bool(has_table),
                "table_date_key": table_date_key,
                "table_delta_days": table_delta_days,
            })

        valid.sort(key=lambda d: (d["ptid"], d["date_key"], d["folder"]))
        print(
            f"[MRI2PETTableDataset] 有效样本数量: {len(valid)} "
            f"(match_policy={self.match_policy}, allow_unmatched_table={self.allow_unmatched_table})"
        )
        return valid

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, index: int) -> Optional[Dict]:
        try:
            sample = self.valid_samples[index]

            if not (Path(sample["mri_path"]).exists() and Path(sample["pet_path"]).exists()):
                return None

            batch = self.transform({"image": sample["mri_path"], "label": sample["pet_path"]})

            if batch["image"].shape[0] != 1 or batch["label"].shape[0] != 1:
                return None

            # ===== 表格特征 & 缺失掩码（拆成两个 key）=====
            F = len(self.tab_stats.columns)

            if sample["has_table"]:
                row = self._table_index.get((sample["ptid"], sample["table_date_key"]), None)
                if row is None:
                    # nearest 的理论兜底
                    row, _, _ = self._match_table(sample["ptid"], sample["date_key"])

                x, miss = _get_tabular_vector(
                    row=row,
                    stats=self.tab_stats,
                    add_missing_mask=self.add_missing_mask,  # True 时 miss 为 1=缺失 0=不缺失
                    standardize=self.standardize_table,
                )
            else:
                # 无表格：table 用 0 向量，mask 全 1（表示全缺失）
                x = np.zeros((F,), dtype=np.float32)
                miss = np.ones((F,), dtype=np.float32) if self.add_missing_mask else None

            batch["table"] = torch.from_numpy(x.astype(np.float32))  # (F,)

            if self.add_missing_mask:
                # 若 has_table=True：miss 来自 np.isnan；若 has_table=False：miss 全 1
                if miss is None:
                    # 理论上不会发生；做兜底
                    miss = np.zeros((F,), dtype=np.float32)
                batch["mask"] = torch.from_numpy(miss.astype(np.float32))  # (F,) 1=缺失 0=不缺失

            # ===== meta 信息保持不变 =====
            batch["folder"] = sample["folder"]
            batch["ptid"] = sample["ptid"]
            batch["date_key"] = sample["date_key"]
            batch["has_table"] = sample["has_table"]
            batch["table_date_key"] = sample["table_date_key"]
            batch["table_delta_days"] = sample["table_delta_days"]
            return batch

        except Exception as e:
            print(f"[MRI2PETTableDataset] 加载失败 idx={index}: {e}")
            return None


def safe_collate(batch):
    """过滤 __getitem__ 返回 None 的样本，避免 DataLoader 崩溃。"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def mri2pet_table_dataloader(
    data_path: Union[str, Path],
    table_csv: Union[str, Path],
    desired_shape: Tuple[int, int, int] = (160, 160, 96),
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    便捷构造 DataLoader。

    dataset_kwargs 会透传给 MRI2PETTableDataset，例如：
      - table_cols=[...]
      - add_missing_mask=True/False
      - standardize_table=True/False
      - tabular_stats=TabularStats 或 dict
      - allow_unmatched_table=True/False
      - strict_channel_check=True/False
      - match_policy="exact" / "nearest"
      - max_date_delta_days=180
    """
    ds = MRI2PETTableDataset(
        data_path=data_path,
        table_csv=table_csv,
        desired_shape=desired_shape,
        **dataset_kwargs,
    )
    print(f"[mri2pet_table_dataloader] 总有效样本数: {len(ds)}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=safe_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
