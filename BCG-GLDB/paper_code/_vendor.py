"""Expose the archived building blocks without making their old entries authoritative."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATION_VENDOR = ROOT / "vendor" / "generation"
CLASSIFICATION_VENDOR = ROOT / "vendor" / "classification"

for directory in (GENERATION_VENDOR, CLASSIFICATION_VENDOR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
