import torch
import torch.nn as nn
from typing import Optional
from .blocks import ResNetBlockUp, SelfAttention2d


class ConditionalDecoder(nn.Module):
    """
    ResNet-based Generator / Decoder.
    Generates realistic images from latent vector z and conditioning class labels.
    Uses Conditional Batch Normalization for smooth attribute injection.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 10,
        condition_dim: int = 64,
        target_size: int = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.target_size = target_size

        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, condition_dim)
            in_dim = latent_dim + condition_dim
        else:
            self.class_embedding = None
            in_dim = latent_dim

        # Initial projection: latent -> [B, base_channels * 4, 4, 4]
        self.init_channels = base_channels * 4
        self.fc_in = nn.Linear(in_dim, self.init_channels * 4 * 4)

        # Upsampling stages: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.block1 = ResNetBlockUp(self.init_channels, base_channels * 4, condition_dim=condition_dim, upsample=True) # 4 -> 8
        self.block2 = ResNetBlockUp(base_channels * 4, base_channels * 2, condition_dim=condition_dim, upsample=True) # 8 -> 16
        self.attn = SelfAttention2d(base_channels * 2)
        self.block3 = ResNetBlockUp(base_channels * 2, base_channels, condition_dim=condition_dim, upsample=True)     # 16 -> 32

        # Final head
        self.final_norm = nn.BatchNorm2d(base_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z: Latent vectors [B, latent_dim]
            labels: Integer class labels [B] or None
        Returns:
            Reconstructed/generated image [B, out_channels, target_size, target_size] in [-1, 1]
        """
        B = z.size(0)

        if self.class_embedding is not None and labels is not None:
            c_emb = self.class_embedding(labels)
            z_in = torch.cat([z, c_emb], dim=1)
        else:
            c_emb = None
            z_in = z

        h = self.fc_in(z_in).view(B, self.init_channels, 4, 4)
        h = self.block1(h, c_emb)
        h = self.block2(h, c_emb)
        h = self.attn(h)
        h = self.block3(h, c_emb)

        h = self.act(self.final_norm(h))
        out = self.tanh(self.final_conv(h))

        return out
