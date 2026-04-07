# config/unet_baseline_1.py
# Baseline experiment config:
# - UNet baseline (no BSFE, no table conditioning)
# - MRI is encoded (downsampled) and concatenated with PET latent/noisy input.
#
# Only edit: paths + save_dir.

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Paths (EDIT THESE)
# -------------------------
train_data = r"/zjs/MRI2PET/dataset/MRI_PET_Table/train"
val_data  = r"/zjs/MRI2PET/dataset/MRI_PET_Table/val"
table_csv = "/zjs/MRI2PET/dataset/table.csv"
test_data  = r"/zjs/MRI2PET/dataset/MRI_PET_Table/test"
save_dir   = r"/zjs/MRI2PET/MRI2PET/result/ddpm/unet_dual_v3"


# optional DVAE checkpoint (only needed if use_pet_latent_encoder=True)
use_pet_latent_encoder = True
vae_dir = r"/zjs/MRI2PET/MRI2PET/result/dvae/best_model.pth"

# -------------------------
# Data
# -------------------------
desired_shape = (160, 160, 96)
add_missing_mask = True
standardize_table = True
allow_unmatched_table = False
strict_channel_check = True

# -------------------------
# Training
# -------------------------

epochs = 500
batch_size = 2
val_batch_size = 1
numworker = 4
learning_rate = 2e-4
amp = True

val_interval = 1
val_mode = "final"
val_max_batches = -1   # -1 => ALL val batches

# -------------------------
# Diffusion
# -------------------------
T = 1000
beta_start = 1e-4
beta_end = 2e-2
schedule = "cosine"
loss_weighting = "p2"
p2_gamma = 0.5
p2_k = 1.0

# dual head
dual_pred = False
dual_out_order = "v_eps"   # 或 "eps_v"

# dual-target loss
use_dual_target_loss = True
loss_v_weight = 1.0
loss_eps_weight = 1.0



cond_drop_prob = 0.0
cfg_scale = 1.0

ddim_steps = 50
ddim_eta = 0.0

clip_x0 = False
psnr_max_value = 2.5

use_ema = True
ema_decay = 0.9999

# -------------------------
# UNet baseline architecture
# -------------------------
latent_dim = 1
cond_channels = 1
mri_in_channels = 1
mri_down = 2
mri_base_channels = 1

inner_channel = 32
norm_groups = 16
channel_mults = (2, 4, 8, 16)
attn_res = (4,)
res_blocks = 3
dropout = 0.1

image_size = 40
time_dim = 128

val_ddim_steps = 1

table_token_dim = 128
table_cross_attn_heads = 4
use_table_cross_attn = True
table_cross_attn_res = (20, 10)

# --- BSFE (replace MRI encoder) ---
use_bsfe = True
bsfe_freeze = True
bsfe_dir = "/zjs/MRI2PET/MRI2PET/result/bsfe/bsfe_only_best.pth"

# 必须与你训练 BSFE 的结构一致（你 bsfe_pretrain_config.py 里那套）
bsfe_input_shape = (160, 160, 96)
bsfe_latent_shape = (40, 40, 24)
bsfe_in_channels = 1
bsfe_pre_down_hidden = 8

bsfe_out_context_dim = 128

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

test_ddim_steps = 5
clip_x0 = False

