"""Exp031 data entry points.

Exp031 intentionally keeps the *exact* Exp027/029/030 data protocol
(split csv, seed, preprocessing, cache, Parc98 labelmap, brain mask) so that
validation and the official evaluation protocol stay comparable.  The dataset
object is re-exported directly from exp027.data.build_dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from exp027.data import (
    Exp027Dataset,
    build_dataset,
    load_labelmap,
    resolve_path,
)

__all__ = ["Exp027Dataset", "build_dataset", "load_labelmap", "resolve_path"]
