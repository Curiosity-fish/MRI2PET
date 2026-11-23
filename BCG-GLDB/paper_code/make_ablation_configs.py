"""Generate the eight module combinations and the final ROI configuration."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


CONFIGURATIONS = {
    "baseline": (False, False, False, False),
    "mri_structure": (True, False, False, False),
    "clinical_table": (False, True, False, False),
    "global_context": (False, False, True, False),
    "mri_clinical": (True, True, False, False),
    "mri_global": (True, False, True, False),
    "clinical_global": (False, True, True, False),
    "three_branches": (True, True, True, False),
    "final_with_parc98": (True, True, True, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, enabled in CONFIGURATIONS.items():
        cfg = deepcopy(source)
        for key, value in zip(("use_mri_structure", "use_clinical", "use_global", "use_roi"), enabled):
            cfg["model"][key] = value
        cfg["output_dir"] = str(Path(source["output_dir"]) / name)
        target = args.output_dir / f"{name}.yaml"
        target.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
