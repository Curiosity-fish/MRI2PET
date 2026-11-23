from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F

from utils.region_crop import CropSlices, compute_crop_slices_from_mask


@dataclass(frozen=True)
class AtlasRegion:
    region_id: int
    region_name: str
    zh_name: str
    mask_path: Path
    voxel_count: int
    bbox_shape: tuple[int, int, int]


class RegionAtlas:
    def __init__(
        self,
        labelmap_path: str | Path,
        lut_path: str | Path,
        bbox_path: Optional[str | Path] = None,
    ) -> None:
        self.labelmap_path = Path(labelmap_path)
        self.lut_path = Path(lut_path)
        self.bbox_path = Path(bbox_path) if bbox_path else self.lut_path.with_name(self.lut_path.stem.replace("_lut", "_bboxes") + ".json")

        self.label_img = nib.load(str(self.labelmap_path))
        self.affine = self.label_img.affine.copy()
        self.header = self.label_img.header.copy()
        self._shape = tuple(int(v) for v in self.label_img.shape[:3])

        lut_df = pd.read_csv(self.lut_path)
        bbox_json = json.loads(self.bbox_path.read_text(encoding="utf-8"))
        self.regions: dict[int, AtlasRegion] = {}
        self._mask_cache: dict[int, torch.Tensor] = {}
        for row in lut_df.to_dict("records"):
            rid = int(row["region_id"])
            bbox = bbox_json[str(rid)]["bbox"]
            self.regions[rid] = AtlasRegion(
                region_id=rid,
                region_name=str(row["region_name"]),
                zh_name=str(row["zh_name"]),
                mask_path=Path(row["mask_path"]),
                voxel_count=int(bbox["voxel_count"]),
                bbox_shape=tuple(int(v) for v in bbox["shape"]),
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def region_ids(self) -> list[int]:
        return sorted(self.regions)

    def get_region(self, region_id: int) -> AtlasRegion:
        return self.regions[int(region_id)]

    def _load_mask(self, region_id: int) -> torch.Tensor:
        region = self.get_region(region_id)
        img = nib.load(str(region.mask_path))
        mask = torch.from_numpy(img.get_fdata(dtype="float32")).unsqueeze(0).float()
        return mask

    def get_region_mask(self, region_id: int, target_shape: Optional[tuple[int, int, int]] = None) -> torch.Tensor:
        region_id = int(region_id)
        if region_id not in self._mask_cache:
            self._mask_cache[region_id] = self._load_mask(region_id)
        mask = self._mask_cache[region_id]
        if target_shape is None or tuple(mask.shape[-3:]) == tuple(target_shape):
            return mask.clone()
        resized = F.interpolate(mask.unsqueeze(0), size=target_shape, mode="nearest").squeeze(0)
        return resized

    def get_crop_slices(
        self,
        region_id: int,
        margin: tuple[int, int, int],
        min_size: tuple[int, int, int],
        pad_to_multiple: int,
        target_shape: Optional[tuple[int, int, int]] = None,
    ) -> CropSlices:
        mask = self.get_region_mask(region_id, target_shape=target_shape)
        return compute_crop_slices_from_mask(mask, margin=margin, min_size=min_size, pad_to_multiple=pad_to_multiple)
