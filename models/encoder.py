import torch
import torch.nn as nn
from typing import Tuple, Optional
from .blocks import ResNetBlockDown, SelfAttention2d


class ConditionalEncoder(nn.Module):
    """
    ResNet-based Conditional Variational Autoencoder Encoder.
    Encodes an input image and condition into latent Gaussian parameters (mu, logvar).
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 64,
        num_classes: int = 10,
        condition_dim: int = 32,
        image_size: int = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.image_size = image_size

        # Class embedding projected to spatial dimensions
        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, condition_dim)
            self.cond_proj = nn.Linear(condition_dim, image_size * image_size)
            conv_in = in_channels + 1
        else:
            self.class_embedding = None
            conv_in = in_channels

        # Initial convolution
        self.in_conv = nn.Conv2d(conv_in, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling stages (e.g. 32 -> 16 -> 8 -> 4)
        self.block1 = ResNetBlockDown(base_channels, base_channels * 2, downsample=True) # 32 -> 16
        self.attn = SelfAttention2d(base_channels * 2)
        self.block2 = ResNetBlockDown(base_channels * 2, base_channels * 4, downsample=True) # 16 -> 8
        self.block3 = ResNetBlockDown(base_channels * 4, base_channels * 4, downsample=True) # 8 -> 4

        # Final spatial resolution is 4x4
        flatten_dim = (base_channels * 4) * 4 * 4

        # Linear projections for mean and log-variance
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [B, C, H, W] in [-1, 1]
            labels: Integer class labels [B] or None
        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        B, C, H, W = x.size()

        if self.class_embedding is not None and labels is not None:
            c_emb = self.class_embedding(labels) # [B, condition_dim]
            c_map = self.cond_proj(c_emb).view(B, 1, H, W)
            x_in = torch.cat([x, c_map], dim=1)
        else:
            x_in = x

        h = self.in_conv(x_in)
        h = self.block1(h)
        h = self.attn(h)
        h = self.block2(h)
        h = self.block3(h)

        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std, where eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
