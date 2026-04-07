import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_3d(img1, img2, border=0, max_value=255.0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # 切片处理
    slices = tuple(slice(border, dim - border) for dim in img1.shape[-3:])
    img1 = img1[..., slices[0], slices[1], slices[2]]
    img2 = img2[..., slices[0], slices[1], slices[2]]

    # 计算MSE
    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    # 使用max_value替代固定值255
    return 10 * math.log10(max_value ** 2 / mse)


if __name__ == '__main__':
    import torch

    x1 = torch.rand(1, 1, 64, 64, 64)
    x2 = x1.clone()
    x1 = torch.rand(1, 1, 64, 64, 16)
    x2 = torch.rand(1, 1, 64, 64, 16)
    psnr = calculate_psnr_3d(x1, x2, max_value=1)
    print(psnr)
