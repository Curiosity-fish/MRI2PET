"""Synthetic smoke test for the final bridge model; no server data required."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.exp037E.exp037e_global_bsfe_unet3d import Exp037EGlobalBSFEUNet3D


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Exp037EGlobalBSFEUNet3D(
        table_num_features=25, feature_type_ids=[0] * 25,
        in_channels=4, out_channels=4, channels=(96, 192, 384, 768),
        time_dim=384, norm_groups=8, dropout=0.0,
        table_token_dim=128, table_depth=2, table_heads=4, table_ff_mult=4,
        table_dropout=0.0, table_num_feature_types=5, bica_heads=4,
        bica_out_init_std=0.005, bsfe_channels=(48, 96, 192, 384),
        global_vit_enabled=True, global_token_dim=256, global_vit_depth=3,
        global_vit_heads=8, global_vit_mlp_ratio=4.0,
        global_vit_dropout=0.0, global_vit_attention_dropout=0.0,
        global_fusion_gate_bias=-2.0, global_fusion_init_alpha=0.1,
    ).to(device).eval()
    x = torch.randn(1, 4, 40, 48, 40, device=device)
    t = torch.tensor([0.5], device=device)
    table = torch.randn(1, 25, device=device)
    mask = torch.ones(1, 25, device=device)
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, t, table, mask, bsfe_pyramid=None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    assert tuple(out.shape) == tuple(x.shape), (out.shape, x.shape)
    assert bool(torch.isfinite(out.float()).all())
    print(f"synthetic_forward_ok device={device} shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
