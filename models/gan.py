import torch
import torch.nn as nn
from typing import Tuple, Optional
from .decoder import ConditionalDecoder
from .discriminator import ProjectionDiscriminator


class ConditionalGAN(nn.Module):
    """
    Baseline Conditional Generative Adversarial Network (cGAN).
    Consists of a Conditional Generator and Projection Discriminator.
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 64,
        num_classes: int = 10,
        condition_dim: int = 64,
        image_size: int = 32
    ):
        super().__init__()
        self.generator = ConditionalDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            condition_dim=condition_dim,
            target_size=image_size
        )
        self.discriminator = ProjectionDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            image_size=image_size
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate images from latent noise z and conditions.
        """
        return self.generator(z, labels)

    def sample(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        if labels is not None:
            labels = labels.to(device)
        return self.generator(z, labels)
