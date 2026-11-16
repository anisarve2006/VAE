import torch
import torch.nn as nn
from typing import Tuple, Optional
from .encoder import ConditionalEncoder
from .decoder import ConditionalDecoder


class ConditionalVAE(nn.Module):
    """
    Baseline Conditional Variational Autoencoder (CVAE).
    Jointly optimizes reconstruction fidelity and KL latent regularization.
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
        self.encoder = ConditionalEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_classes=num_classes,
            condition_dim=condition_dim,
            image_size=image_size
        )
        self.decoder = ConditionalDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            condition_dim=condition_dim,
            target_size=image_size
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CVAE.
        Returns:
            recon_x: Reconstructed images
            mu: Latent distribution mean
            logvar: Latent distribution log-variance
            z: Sampled latent code
        """
        mu, logvar = self.encoder(x, labels)
        z = self.encoder.reparameterize(mu, logvar)
        recon_x = self.decoder(z, labels)
        return recon_x, mu, logvar, z

    def sample(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Sample new images from the standard normal prior p(z) ~ N(0, I).
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        if labels is not None:
            labels = labels.to(device)
        return self.decoder(z, labels)
