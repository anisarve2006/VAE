import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization layer.
    Modulates scale (gamma) and shift (beta) based on condition embedding.
    """
    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Linear(condition_dim, num_features)
        self.beta_embed = nn.Linear(condition_dim, num_features)

        # Initialize to identity modulation
        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.gamma_embed.bias)
        nn.init.zeros_(self.beta_embed.weight)
        nn.init.zeros_(self.beta_embed.bias)

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.gamma_embed(c_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_embed(c_emb).unsqueeze(-1).unsqueeze(-1)
        return out * gamma + beta


class SelfAttention2d(nn.Module):
    """
    Self-Attention block for 2D feature maps (SAGAN-style).
    Allows generator and discriminator to model non-local dependencies.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, c, h, w = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, h * w).permute(0, 2, 1) # [B, HW, C/8]
        proj_key = self.key_conv(x).view(batch_size, -1, h * w)                       # [B, C/8, HW]
        energy = torch.bmm(proj_query, proj_key)                                       # [B, HW, HW]
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(batch_size, -1, h * w)                   # [B, C, HW]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, c, h, w)

        return self.gamma * out + x


class ResNetBlockDown(nn.Module):
    """
    Residual Downsampling Block for Encoders and Discriminators.
    Optionally applies Spectral Normalization for GAN stability.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, use_spectral_norm: bool = False):
        super().__init__()
        self.downsample = downsample
        conv = (lambda c_in, c_out, k, s, p: spectral_norm(nn.Conv2d(c_in, c_out, k, s, p))) if use_spectral_norm else nn.Conv2d

        self.conv1 = conv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = conv(out_channels, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        if in_channels != out_channels or downsample:
            self.shortcut = conv(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            res = F.avg_pool2d(res, 2)
        res = self.shortcut(res)
        return self.act(h + res)


class ResNetBlockUp(nn.Module):
    """
    Residual Upsampling Block for Decoders and Generators.
    Supports Conditional Batch Normalization and PixelShuffle or Nearest Upsampling.
    """
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int = 0, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.use_cond = condition_dim > 0

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if self.use_cond:
            self.norm1 = ConditionalBatchNorm2d(in_channels, condition_dim)
            self.norm2 = ConditionalBatchNorm2d(out_channels, condition_dim)
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        if in_channels != out_channels or upsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor = None) -> torch.Tensor:
        res = x
        if self.upsample:
            res = F.interpolate(res, scale_factor=2, mode="nearest")

        if self.use_cond and c_emb is not None:
            h = self.act(self.norm1(x, c_emb))
        else:
            h = self.act(self.norm1(x))

        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")

        h = self.conv1(h)

        if self.use_cond and c_emb is not None:
            h = self.act(self.norm2(h, c_emb))
        else:
            h = self.act(self.norm2(h))

        h = self.conv2(h)
        res = self.shortcut(res)
        return h + res


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) for Super-Resolution feature extraction.
    Dense connections preserve multi-scale features and gradient flow.
    """
    def __init__(self, channels: int = 64, growth_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) combining multiple RDBs with a residual skip.
    """
    def __init__(self, channels: int = 64, growth_channels: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class PixelShuffleUpsampler(nn.Module):
    """
    Sub-pixel convolution upsampler for 2x or 4x magnification without checkerboard artifacts.
    """
    def __init__(self, in_channels: int, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        if scale_factor == 4:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        elif scale_factor == 2:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
