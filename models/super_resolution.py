import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from torch.nn.utils.parametrizations import spectral_norm
from .blocks import RRDB, PixelShuffleUpsampler, ResNetBlockDown


class SREncoder(nn.Module):
    """
    Encodes low-resolution image into latent distribution (mu, logvar)
    representing the ill-posed high-frequency texture uncertainty.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 64, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.down1 = ResNetBlockDown(base_channels, base_channels * 2, downsample=True)
        self.down2 = ResNetBlockDown(base_channels * 2, base_channels * 4, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(base_channels * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 4, latent_dim)

    def forward(self, x_lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.conv1(x_lr), 0.2)
        h = self.down1(h)
        h = self.down2(h)
        pooled = self.pool(h).flatten(1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SRGenerator(nn.Module):
    """
    High-capacity Super-Resolution Generator with RRDB backbone,
    latent texture modulation, and sub-pixel convolution upsampling.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 64,
        num_rrdb: int = 8,
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_first = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        # Latent modulation layer: expands latent code to spatial feature map
        self.latent_proj = nn.Linear(latent_dim, base_channels)

        # RRDB Trunk
        self.rrdb_blocks = nn.ModuleList([
            RRDB(channels=base_channels, growth_channels=32) for _ in range(num_rrdb)
        ])
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, 1, 1)

        # Upsampling
        self.upsampler = PixelShuffleUpsampler(base_channels, scale_factor=scale_factor)
        self.conv_hr = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x_lr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x_lr)

        # Inject latent prior modulation
        z_emb = self.latent_proj(z).unsqueeze(-1).unsqueeze(-1)
        h = feat + z_emb

        body = h
        for block in self.rrdb_blocks:
            body = block(body)
        body = self.conv_body(body)

        # Global residual skip from shallow LR features
        out = feat + body

        out = self.upsampler(out)
        out = self.act(self.conv_hr(out))
        out = self.tanh(self.conv_last(out))

        return out


class SRDiscriminator(nn.Module):
    """
    Patch-based Discriminator for evaluating Super-Resolution realism.
    Operates at high-resolution with Spectral Normalization.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        conv = lambda c_in, c_out, k, s, p: spectral_norm(nn.Conv2d(c_in, c_out, k, s, p))

        self.net = nn.Sequential(
            conv(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_channels, base_channels, 3, 2, 1), # down 2x
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_channels, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_channels * 2, base_channels * 2, 3, 2, 1), # down 4x
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_channels * 4, base_channels * 4, 3, 2, 1), # down 8x
            nn.LeakyReLU(0.2, inplace=True),

            conv(base_channels * 4, 1, 3, 1, 1) # Patch output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAESRGAN(nn.Module):
    """
    Unified VAE-SRGAN Super-Resolution Framework.
    Solves ill-posed high-frequency synthesis via conditional latent modeling
    and adversarial perceptual optimization.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 64,
        num_rrdb: int = 8,
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_dim = latent_dim

        self.encoder = SREncoder(in_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels)
        self.generator = SRGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            num_rrdb=num_rrdb,
            scale_factor=scale_factor
        )
        self.discriminator = SRDiscriminator(in_channels=out_channels, base_channels=base_channels)

    def forward(self, x_lr: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x_lr)
        z = self.encoder.reparameterize(mu, logvar)
        x_sr = self.generator(x_lr, z)
        return {
            "x_sr": x_sr,
            "mu": mu,
            "logvar": logvar,
            "z": z
        }

    def super_resolve(self, x_lr: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Inference super-resolution function.
        """
        mu, logvar = self.encoder(x_lr)
        z = mu if deterministic else self.encoder.reparameterize(mu, logvar)
        return self.generator(x_lr, z)
