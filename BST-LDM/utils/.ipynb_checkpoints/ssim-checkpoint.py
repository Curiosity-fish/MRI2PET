import torch
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma, device=None, dtype=None):
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        device=device, dtype=dtype
    )
    return gauss / gauss.sum()

def create_window_3D(window_size, channel, device=None, dtype=torch.float32):
    _1D_window = gaussian(window_size, 1.5, device=device, dtype=dtype).unsqueeze(1)  # (W,1)
    _2D_window = _1D_window @ _1D_window.t()                                          # (W,W)
    _3D_window = (_1D_window @ _2D_window.reshape(1, -1)).reshape(
        window_size, window_size, window_size
    ).unsqueeze(0).unsqueeze(0)                                                       # (1,1,W,W,W)

    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3, 4))

class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = None
        self.register_buffer("window", torch.empty(0))  # buffer: 自动跟 device / state_dict

    def forward(self, img1, img2):
        # 统一用 float32（强烈推荐）
        img1 = img1.float()
        img2 = img2.float().to(dtype=img1.dtype, device=img1.device)

        (_, channel, _, _, _) = img1.size()

        if self.channel != channel or self.window.numel() == 0 or self.window.device != img1.device:
            self.window = create_window_3D(self.window_size, channel, device=img1.device, dtype=img1.dtype)
            self.channel = channel
        else:
            # 确保 dtype 一致（以防万一）
            if self.window.dtype != img1.dtype:
                self.window = self.window.to(dtype=img1.dtype)

        return _ssim_3D(img1, img2, self.window, self.window_size, channel, self.size_average)

def ssim3D(img1, img2, window_size=11, size_average=True):
    # 统一用 float32（强烈推荐）
    img1 = img1.float()
    img2 = img2.float().to(dtype=img1.dtype, device=img1.device)

    (_, channel, _, _, _) = img1.shape
    window = create_window_3D(window_size, channel, device=img1.device, dtype=img1.dtype)
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
