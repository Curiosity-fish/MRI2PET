# model/VAE.py
import torch
from torch import nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups=16):
        super().__init__()
        g = min(num_groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(num_groups=g, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.skip(x) + self.block(x)


class EncoderVAE(nn.Module):
    """
    输出: mu, logvar
    形状: (B, latent_dim, D', H', W')
    """
    def __init__(self, image_channels=1, latent_dim=16):
        super().__init__()
        channels = [8, 16, 16, 32]
        num_res_blocks = 2

        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            if i % 2 == 0:
                layers.append(Downsample(out_ch))

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(nn.ReLU(inplace=True))

        # 输出 2*latent_dim 用于 mu 与 logvar
        layers.append(nn.Conv3d(channels[-1], 2 * latent_dim, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        h = self.model(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, image_channels=1, input_channels=16):
        super().__init__()
        channels = [32, 16, 16, 8]
        num_res_blocks = 2

        in_channels = channels[0]
        layers = [
            nn.Conv3d(input_channels, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i % 2 == 0:
                layers.append(Upsample(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class VAE(nn.Module):
    def __init__(self, image_channels=1, latent_dim=16):
        super().__init__()
        self.encoder = EncoderVAE(image_channels=image_channels, latent_dim=latent_dim)
        self.decoder = Decoder(image_channels=image_channels, input_channels=latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


def vae_loss(x_hat, x, mu, logvar, beta=1.0, recon_type="l1"):
    """
    loss = recon + beta * KL
    recon: mean over all elements
    KL: mean over all elements
    """
    if recon_type.lower() == "l2":
        recon = F.mse_loss(x_hat, x, reduction="mean")
    else:
        recon = F.l1_loss(x_hat, x, reduction="mean")

    kl_map = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_map.mean()

    loss = recon + beta * kl
    return loss, recon, kl

class _GN(nn.Module):
    """GroupNorm helper: auto adjust groups to be divisible."""
    def __init__(self, c: int, num_groups: int = 16):
        super().__init__()
        g = min(num_groups, c)
        while c % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, c, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Discriminator3DPatch(nn.Module):
    """
    3D PatchGAN Discriminator
    Input : (B, C, D, H, W)
    Output: (B, 1, d', h', w') logits (no sigmoid)
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        max_channels: int = 256,
        norm: str = "gn",  # "gn" or "none"
    ):
        super().__init__()

        layers = []
        ch = base_channels

        # first layer: no norm
        layers += [
            nn.Conv3d(in_channels, ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for _ in range(1, num_layers):
            ch_next = min(ch * 2, max_channels)
            layers.append(nn.Conv3d(ch, ch_next, kernel_size=4, stride=2, padding=1))
            if norm == "gn":
                layers.append(_GN(ch_next))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = ch_next

        # final: keep spatial, output patch logits
        layers.append(nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



if __name__ == "__main__":
    model = VAE(image_channels=1, latent_dim=16).cuda()
    x = torch.randn(1, 1, 80, 160, 160).cuda()  # (N,C,D,H,W) 示例
    x_hat, mu, logvar, z = model(x)
    print("x:", x.shape)
    print("x_hat:", x_hat.shape, "mu:", mu.shape, "logvar:", logvar.shape, "z:", z.shape)
