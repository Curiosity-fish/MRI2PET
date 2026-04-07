import torch
import os

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")


latent_dim = 1
learning_rate = 1e-4
batch_size = 6
numworker = 4
epochs = 100
model_name = 'dvae_l1'
train_data = '/zjs/MRI2PET/dataset/MRI_PET_split/train'
test_data = '/zjs/MRI2PET/dataset/MRI_PET_split/test'

save_dir = '/zjs/MRI2PET/MRI2PET/result/dvae/dvae_l1_v2'

amp = True
recon_type = "l1"
ssim_weight = 0.5
psnr_weight = 0.01
beta_max = 1e-2
beta_warmup_epochs = 20
psnr_max_value = 2.5

use_gan = True
lambda_adv_max = 1e-3       
adv_warmup_epochs = 10
d_steps = 1

pet_normalize = False
pet_clip_max = 2.5          

beta_max = 1e-2
beta_warmup_epochs = 20
ssim_weight = 0.5
recon_type = "l1"
amp = True

d_base_channels = 32
d_layers = 4
