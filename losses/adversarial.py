import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """
    Adversarial loss computation supporting Hinge, BCE, and Relativistic Average formulations.
    """
    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates loss for updating Discriminator.
        """
        if self.loss_type == "hinge":
            # D_loss = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
            loss_real = torch.mean(F.relu(1.0 - real_pred))
            loss_fake = torch.mean(F.relu(1.0 + fake_pred))
            return loss_real + loss_fake

        elif self.loss_type == "bce":
            loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
            loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            return (loss_real + loss_fake) * 0.5

        elif self.loss_type == "relativistic":
            # RaGAN Discriminator loss
            diff_real = real_pred - torch.mean(fake_pred)
            diff_fake = fake_pred - torch.mean(real_pred)
            loss_real = F.binary_cross_entropy_with_logits(diff_real, torch.ones_like(real_pred))
            loss_fake = F.binary_cross_entropy_with_logits(diff_fake, torch.zeros_like(fake_pred))
            return (loss_real + loss_fake) * 0.5

        else:
            raise ValueError(f"Unknown adversarial loss type: {self.loss_type}")

    def generator_loss(
        self,
        fake_pred: torch.Tensor,
        real_pred: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculates loss for updating Generator to fool Discriminator.
        """
        if self.loss_type == "hinge":
            # G_loss = -E[D(G(z))]
            return -torch.mean(fake_pred)

        elif self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

        elif self.loss_type == "relativistic":
            if real_pred is None:
                raise ValueError("Relativistic GAN generator loss requires real_pred.")
            diff_real = real_pred - torch.mean(fake_pred)
            diff_fake = fake_pred - torch.mean(real_pred)
            loss_real = F.binary_cross_entropy_with_logits(diff_real, torch.zeros_like(diff_real))
            loss_fake = F.binary_cross_entropy_with_logits(diff_fake, torch.ones_like(diff_fake))
            return (loss_real + loss_fake) * 0.5

        else:
            raise ValueError(f"Unknown adversarial loss type: {self.loss_type}")
