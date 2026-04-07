
# bsfe_pretrain_config.py
# Pretrain BSFE by learning MRI -> PET latent (via frozen PET VAE encoder).
# Shapes:
#   MRI: (1, 96, 160, 160)
#   PET latent: (latent_dim, 24, 40, 40)

import os
import torch

# -------------------------
# DDP device
# -------------------------
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

# -------------------------
# paths (edit these)
# -------------------------
train_data = "/zjs/MRI2PET/dataset/MRI_PET_split/train"
test_data = "/zjs/MRI2PET/dataset/MRI_PET_split/test"
table_csv = "/zjs/MRI2PET/dataset/table.csv"   # optional; dataset needs it if your loader requires
save_dir = "/zjs/MRI2PET/MRI2PET/result/bsfe_v3_1"

# PET VAE (frozen) used to produce latent targets
use_pet_latent_encoder = True
vae_dir = "/zjs/MRI2PET/MRI2PET/result/dvae/dvae_l1/best_model.pth"
latent_dim = 1

# -------------------------
# dataset
# -------------------------
desired_shape = (160, 160, 96)  # your loader uses (H,W,D) or (D,H,W); keep consistent with your v2 scripts
table_cols = None
add_missing_mask = True
standardize_table = True
allow_unmatched_table = False
match_policy = "nearest"
max_date_delta_days = 30
strict_channel_check = True

# -------------------------
# BSFE model
# -------------------------
# BSFE runs in latent coordinates after pre-downsample:
bsfe_base_channels = 32
bsfe_channel_mults = (2, 4, 8, 16)
bsfe_num_res_blocks = 3
bsfe_norm_groups = 32
bsfe_dropout = 0.1
bsfe_attn_stages = (2, 3)
bsfe_attn_heads = 4
bsfe_attn_head_dim = 32
bsfe_n_tokens = 32
bsfe_token_dim = 128
bsfe_global_dim = 256
bsfe_out_context_dim = None  # set to diffusion context dim if you want tokens pre-projected

# Head model
head_fusion_ch = 64
head_blocks = 2
head_use_global_film = True
pet_target_spatial = desired_shape


# -------------------------
# training
# -------------------------
epochs = 500
batch_size = 2
val_batch_size = 2
learning_rate = 1e-4
weight_decay = 1e-4
numworker = 4
amp = True

# loss weights
w_latent_l1 = 1.0
w_latent_grad = 0.1
w_pet_l1 = 1.0       # optional: decode z_hat -> PET_hat and compute L1 in PET space
w_pet_ssim = 0.1     # optional: decode and compute SSIM3D (if you have ssim3D util)

# validation / saving
val_interval = 1
save_interval = 1
