import os
import torch
import torch.distributed as dist

import config.unet as config
from train.train_unet import train_DDPM


def ddp_setup():
    """
    torchrun 会自动设置环境变量：RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    is_ddp, local_rank = ddp_setup()

    # 每个进程绑定自己的 GPU
    if torch.cuda.is_available():
        config.device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        config.device = torch.device("cpu")

    train_DDPM(config)

    ddp_cleanup()


if __name__ == "__main__":
    main()
