import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from .encoder import ConditionalEncoder
from .decoder import ConditionalDecoder
from .discriminator import ProjectionDiscriminator


class ConditionalVAEGAN(nn.Module):
    """
    Unified Conditional VAE-GAN Hybrid Framework.
    Combines VAE latent-space regularity and continuous manifold interpolation
    with GAN perceptual sharpness and adversarial feedback.
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
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.image_size = image_size

        # Encoder: E_theta(x, c) -> (mu, logvar)
        self.encoder = ConditionalEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_classes=num_classes,
            condition_dim=condition_dim,
            image_size=image_size
        )

        # Decoder / Generator: G_theta(z, c) -> x_hat
        self.generator = ConditionalDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            condition_dim=condition_dim,
            target_size=image_size
        )

        # Discriminator: D_phi(x, c) -> score and intermediate feature representations
        self.discriminator = ProjectionDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            image_size=image_size
        )

    def encode(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, labels)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return self.encoder.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.generator(z, labels)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass performing both reconstruction and prior sampling.
        Returns:
            Dictionary with reconstructed images, prior samples, latent distributions, and samples.
        """
        B = x.size(0)

        # 1. Posterior path (Reconstruction)
        mu, logvar = self.encode(x, labels)
        z_rec = self.reparameterize(mu, logvar)
        x_rec = self.decode(z_rec, labels)

        # 2. Prior path (Generation from pure Gaussian noise)
        z_prior = torch.randn(B, self.latent_dim, device=x.device)
        x_prior = self.decode(z_prior, labels)

        return {
            "x_rec": x_rec,
            "x_prior": x_prior,
            "mu": mu,
            "logvar": logvar,
            "z_rec": z_rec,
            "z_prior": z_prior
        }

    def sample(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Generates novel conditional images from the prior distribution.
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        if labels is not None:
            labels = labels.to(device)
        return self.generator(z, labels)

    def reconstruct(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Direct autoencoding reconstruction for test/evaluation.
        """
        mu, logvar = self.encode(x, labels)
        z = mu # Use mean directly for deterministic evaluation
        return self.decode(z, labels)
