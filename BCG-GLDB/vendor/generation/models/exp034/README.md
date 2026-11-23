# Exp034 Source Bundle

This directory collects the source code and configurations used by the Exp034
series. It is a source bundle, not a self-contained data/checkpoint archive.

## Contents

- `configs/`: Exp034A-N configurations, including `Exp034M_v3` and `Exp034N01`.
- `models/`: Exp034 U-Net, attention, table, BSFE, Mamba and fusion modules.
- `train/`: Exp034 experiment entry points.
- `ae/`: patch AE/VAE training configurations, AE training scripts, the patch
  latent AE implementation, and the loader used by later bridge experiments.
- `dependencies/`: local patch codec, table interaction, Mamba and BSFE builder
  sources required by the copied training code.
- `docs/`: detailed Exp034M_v3 architecture and training description.

## Main experiments

### Exp034M_v3

```text
configs/t1mri_fdgpet_exp034m_v3_joint_full.yaml
train/train_exp034m_v2_curriculum.py
models/exp034m_curriculum_unet3d.py
```

The script is named `v2` for historical compatibility; the v3 configuration
uses its `--all-active` mode to enable table, BSFE, Mamba, PET decoder, ROI and
SSIM paths from epoch 1.

### Exp034N01

```text
configs/t1mri_fdgpet_exp034n01_stage1_overfit.yaml
train/train_exp034m_curriculum.py --stage1-only
```

N01 is the independently tuned stage-1 branch. Its configuration sets the
additional deepest bottleneck to 1024 channels.

## AE checkpoints

The bridge configurations reference existing checkpoints outside this bundle:

```text
PET AE : experiments/Exp033J_AE_patch_pet_full/checkpoints/best.pt
MRI AE : experiments/Exp034B_AE_patch_mri_full/checkpoints/best.pt
BSFE   : experiments/Exp011b_bsfe_t1mri_fdgpet_fold0/checkpoints/fold0_best.pt
```

These large binary files are intentionally not copied. Training data, atlas
files, cache files and experiment outputs are also intentionally not copied.

## Important path note

The copied scripts retain the original repository imports (`exp026`, `exp032e`,
`models`, `train`, and dataset modules). Run them from the repository root or
set `PYTHONPATH` to the repository root. The bundle preserves source and
configuration provenance; it does not rewrite all imports into a standalone
Python package.

## Verification

The source files were copied from the working repository after the Exp034M_v3
and Exp034N01 implementations. Use the original training commands from the
repository root so data paths and checkpoint paths resolve consistently.
