import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from torch.nn.utils.parametrizations import spectral_norm
from .blocks import ResNetBlockDown, SelfAttention2d


class ProjectionDiscriminator(nn.Module):
    """
    ResNet-based Projection Discriminator with Spectral Normalization.
    Extracts intermediate features for Feature Matching Loss and uses
    inner-product projection for conditional discrimination (Miyato et al.).
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 10,
        image_size: int = 32
    ):
        super().__init__()
        self.num_classes = num_classes

        # Initial Conv
        self.in_conv = spectral_norm(nn.Conv2d(in_channels, base_channels, 3, 1, 1))

        # Downsampling stages: 32 -> 16 -> 8 -> 4
        self.block1 = ResNetBlockDown(base_channels, base_channels * 2, downsample=True, use_spectral_norm=True)
        self.attn = SelfAttention2d(base_channels * 2)
        self.block2 = ResNetBlockDown(base_channels * 2, base_channels * 4, downsample=True, use_spectral_norm=True)
        self.block3 = ResNetBlockDown(base_channels * 4, base_channels * 4, downsample=True, use_spectral_norm=True)

        # Global average pooling + linear head
        self.linear_head = spectral_norm(nn.Linear(base_channels * 4, 1))

        # Conditional projection embedding
        if num_classes > 0:
            self.class_embed = spectral_norm(nn.Embedding(num_classes, base_channels * 4))
        else:
            self.class_embed = None

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input images [B, C, H, W]
            labels: Integer class labels [B] or None
            return_features: If True, returns intermediate activation maps for feature matching
        Returns:
            logits: Adversarial validity scores [B, 1]
            features: List of feature maps if return_features=True else None
        """
        features = []

        h = self.in_conv(x)
        features.append(h)

        h = self.block1(h)
        features.append(h)

        h = self.attn(h)
        h = self.block2(h)
        features.append(h)

        h = self.block3(h)
        features.append(h)

        # Global pooling [B, C]
        h_pool = torch.sum(h, dim=[2, 3])

        # Unconditional score
        out = self.linear_head(h_pool)

        # Conditional projection score
        if self.class_embed is not None and labels is not None:
            c_emb = self.class_embed(labels)
            out = out + torch.sum(h_pool * c_emb, dim=1, keepdim=True)

        if return_features:
            return out, features
        return out, None
