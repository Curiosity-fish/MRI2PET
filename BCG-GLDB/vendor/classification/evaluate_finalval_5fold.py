"""Evaluate one saved classifier on five stratified subsets of its final-val set.

This is a resampling-style uncertainty estimate for one trained checkpoint; it
does not retrain five cross-validation models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
CLASSIER_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MRI2PET.classier.train.train_medicalnet_dx import compute_metrics
from tglm50.train_exp08_table_global_local import Exp08, TableDataset, metrics as tglm_metrics
from MRI2PET.classier import PairedPETDiagnosisDataset, create_medicalnet_classifier


TGLM_CONFIGS = {
    "TGLM50_ours_from_best_bs4_gpu5": "configs/rerun_tglm50_ours_from_best_bs4_gpu5.json",
    "TGLM50_RegGAN_PET_rerun_gpu3": "configs/rerun_tglm50_reggan_pet_gpu3.json",
    "TGLM50_PET_table_global_local_gpu3_seq": "configs/dx3_exp08_pet_gpu3_seq.yaml",
    "TGLM50_unet_tmi_PET_table_global_local": "configs/generated_tglm50_unet_tmi.json",
    "TGLM50_pasta_tmi_PET_table_global_local": "configs/generated_tglm50_pasta_tmi.json",
    "TGLM50_bicyclegan_tmi_PET_table_global_local": "configs/generated_tglm50_bicyclegan_tmi.json",
    "TGLM50_bbdm_PET_table_global_local": "configs/generated_tglm50_bbdm.json",
    "TGLM50_syndiff_PET_table_global_local": "configs/generated_tglm50_syndiff.json",
    "TGLM50_voldit_PET_table_global_local": "configs/generated_tglm50_voldit.json",
    "TGLM50_standard_3d_ldm_PET_table_global_local": "configs/generated_tglm50_standard_3d_ldm.json",
    "TGLM50_ours_PET_gpu2": "configs/ours_tglm50_pet_gpu2.json",
}


def load_config(exp: Path) -> tuple[dict, bool]:
    if exp.name in TGLM_CONFIGS:
        p = CLASSIER_ROOT / TGLM_CONFIGS[exp.name]
        return (json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())), True
    return yaml.safe_load((exp / "config.yaml").read_text()), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--fold-seed", type=int, default=42)
    a = ap.parse_args()
    exp = a.experiment.resolve()
    cfg, is_tglm = load_config(exp)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)

    if is_tglm:
        train = TableDataset(cfg["labels_csv"], cfg["split_csv"], cfg["table_csv"], "train", modality=cfg.get("modality", "pet"), cache_dir=cfg.get("cache_dir"))
        ds = TableDataset(cfg["labels_csv"], cfg["split_csv"], cfg["table_csv"], "final_val", stats=train.stats, modality=cfg.get("modality", "pet"), cache_dir=cfg.get("cache_dir"))
        model = Exp08(in_channels=2 if cfg.get("modality") == "mri_pet" else 1).to(dev)
        ckpt = exp / "best.pt"
    else:
        dc, mc = cfg["data"], cfg["model"]
        ds = PairedPETDiagnosisDataset(labels_csv=dc["labels_csv"], split_csv=dc["split_csv"], split="final_val", fold=None, spatial_size=tuple(dc["spatial_size"]), cache_dir=dc.get("cache_dir"))
        model = create_medicalnet_classifier(model_depth=int(mc["depth"]), n_classes=int(mc["num_classes"]), in_channels=int(mc.get("in_channels", 1)), shortcut_type=str(mc.get("shortcut_type", "A")), dropout=float(mc.get("dropout", 0.0)), head_hidden_dims=[int(x) for x in mc.get("head_hidden_dims", [])], checkpoint_path=None).to(dev)
        ckpt = exp / "checkpoints" / "best.pt"
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    frame = ds.d if is_tglm else ds.df
    labels = np.asarray(frame["label"].astype(int).tolist())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.fold_seed)
    rows = []
    with torch.no_grad():
        for fold, (_, idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            dl = DataLoader(Subset(ds, idx.tolist()), batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
            ys, ps, prob = [], [], []
            for batch in dl:
                x = batch["input"].to(dev, non_blocking=True)
                y = batch["label"].cpu().tolist()
                logits = model(x, batch["table"].to(dev), batch["table_mask"].to(dev)).float() if is_tglm else model(x)
                ys += y
                ps += logits.argmax(1).cpu().tolist()
                prob += torch.softmax(logits, 1).cpu().tolist()
            m = tglm_metrics(ys, ps, prob) if is_tglm else compute_metrics(ys, ps, prob, 3)
            pre = m["macro_precision"] if "macro_precision" in m else m["macro_pre"]
            rec = m["macro_recall"] if "macro_recall" in m else m["macro_rec"]
            rows.append({"fold": fold, "n": len(idx), "accuracy": m["accuracy"], "macro_precision": pre, "macro_recall": rec, "macro_f1": m["macro_f1"], "macro_auc": m["macro_auc"]})
            print(f"{exp.name} fold={fold}/5 n={len(idx)} f1={m['macro_f1']:.4f}", flush=True)
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"]
    summary = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows], ddof=1))} for k in keys}
    out = {"experiment": exp.name, "checkpoint": str(ckpt), "best_epoch": int(state.get("epoch", -1)), "protocol": "5 stratified disjoint subsets of final_val; fixed trained checkpoint; seed=42", "folds": rows, "summary": summary}
    (exp / "fivefold_finalval_metrics.json").write_text(json.dumps(out, indent=2))
    print("SAVED", exp / "fivefold_finalval_metrics.json", flush=True)


if __name__ == "__main__":
    main()
