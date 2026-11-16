import torch
import torch.nn as nn
import math


class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss between variational posterior q(z|x) ~ N(mu, sigma^2)
    and unit isotropic Gaussian prior p(z) ~ N(0, I).
    Includes 'Free Bits' regularization to prevent posterior collapse.
    """
    def __init__(self, free_bits: float = 0.0):
        super().__init__()
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        Returns:
            Scalar KL loss averaged over batch.
        """
        # Element-wise KL divergence for each latent dimension:
        # KL = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()) # [B, latent_dim]

        if self.free_bits > 0.0:
            # Free bits: clamp minimum KL per dimension before summing
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over latent dimensions, average over batch
        kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))
        return kl_loss


class AnnealingScheduler:
    """
    Dynamic beta scheduler for KL loss weight.
    Supports linear warmup and cyclical schedules (Fu et al., 2019).
    """
    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 10,
        max_beta: float = 1.0,
        schedule_type: str = "linear",
        cycles: int = 4
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.max_beta = max_beta
        self.schedule_type = schedule_type
        self.cycles = cycles

    def get_beta(self, current_epoch: int, current_step: int = 0, steps_per_epoch: int = 1) -> float:
        total_steps = self.total_epochs * steps_per_epoch
        step = current_epoch * steps_per_epoch + current_step

        if self.schedule_type == "constant":
            return self.max_beta

        elif self.schedule_type == "linear":
            warmup_steps = self.warmup_epochs * steps_per_epoch
            if step < warmup_steps:
                return self.max_beta * (step / float(max(1, warmup_steps)))
            return self.max_beta

        elif self.schedule_type == "cyclical":
            period = total_steps // self.cycles
            step_in_cycle = step % period
            ratio = step_in_cycle / float(period)
            if ratio < 0.5:
                return self.max_beta * (ratio * 2.0)
            return self.max_beta

        elif self.schedule_type == "cosine":
            progress = min(1.0, step / float(max(1, self.warmup_epochs * steps_per_epoch)))
            return self.max_beta * 0.5 * (1.0 - math.cos(math.pi * progress))

        else:
            return self.max_beta
