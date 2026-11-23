from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.paired_t1_fdg_table_dataset import PairedT1FDGTableDataset


def resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_labelmap(path: Path, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Load an integer Parc98 labelmap and resize it with nearest-neighbor only."""
    if not path.exists():
        raise FileNotFoundError(f"Parc98 labelmap not found: {path}")
    values = nib.load(str(path)).get_fdata(dtype="float32")
    labels = torch.from_numpy(values).long()[None, None]
    if tuple(labels.shape[-3:]) != tuple(target_shape):
        labels = F.interpolate(
            labels.float(), size=target_shape, mode="nearest"
        ).long()
    labels = labels.squeeze(0).contiguous()
    unique = set(int(value) for value in torch.unique(labels).tolist())
    if not unique.issubset(set(range(99))):
        raise ValueError(
            f"Parc98 labelmap contains labels outside 0..98: {sorted(unique)}"
        )
    return labels


class Exp027Dataset(Dataset):
    """Exp025 table dataset plus one fixed, target-independent Parc98 labelmap."""

    def __init__(
        self,
        base: PairedT1FDGTableDataset,
        parc98: torch.Tensor,
    ) -> None:
        self.base = base
        self.parc98 = parc98.float().contiguous()
        self.table_cols = base.table_cols
        self.tabular_stats = base.tabular_stats
        self.df = base.df

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        # The inherited dataset builds its mask from T1 OR PET.  Exp027 must not
        # use target-derived support for either optimization or validation.
        brain_mask = (sample["input"].abs() > 1e-6).float()
        return {
            "image": sample["input"],
            "label": sample["target"],
            "table": sample["table"],
            "mask": sample["table_mask"],
            "brain_mask": brain_mask,
            "parc98": self.parc98,
            "subject_id": sample["subject_id"],
            "sample_id": sample["sample_id"],
        }


def build_dataset(
    config: dict,
    repo_root: Path,
    split: str,
    train_dataset: Optional[Exp027Dataset] = None,
) -> Exp027Dataset:
    data = config["data"]
    target_shape = tuple(int(value) for value in data["image_size"])
    base = PairedT1FDGTableDataset(
        pairs_csv=resolve_path(repo_root, data["pairs_csv"]),
        split_csv=resolve_path(repo_root, data["split_csv"]),
        table_csv=resolve_path(repo_root, data["table_csv"]),
        split=split,
        fold=data.get("fold"),
        spatial_size=target_shape,
        tabular_stats=(train_dataset.tabular_stats if train_dataset else None),
        table_cols=(train_dataset.table_cols if train_dataset else data.get("table_cols")),
        max_samples=data.get(f"{split}_max_samples"),
        standardize_table=bool(data.get("standardize_table", True)),
        add_missing_mask=True,
        cache_dir=resolve_path(repo_root, data.get("cache_dir")),
    )
    labelmap = load_labelmap(
        resolve_path(repo_root, data["parc98_labelmap_path"]), target_shape
    )
    return Exp027Dataset(base, labelmap)
