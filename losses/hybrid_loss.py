import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from .reconstruction import ReconstructionLoss
from .kl_divergence import KLDivergenceLoss
from .adversarial import AdversarialLoss
from .perceptual import VGGPerceptualLoss


class FeatureMatchingLoss(nn.Module):
    """
    Discriminator Feature Matching Loss.
    Measures the L1 difference between intermediate activation representations
    of real and generated images inside the discriminator.
    Stabilizes adversarial dynamics and mitigates mode collapse.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, real_features: List[torch.Tensor], fake_features: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for f_real, f_fake in zip(real_features, fake_features):
            loss += self.l1(f_fake, f_real.detach())
        return loss / max(1, len(real_features))


class HybridVAEGANLoss(nn.Module):
    """
    Composite Multi-Objective Loss for VAE-GAN Hybrid Training.
    Formula:
        L_G = lambda_rec * L_rec + beta * L_KL + lambda_adv * L_adv + lambda_perc * L_perc + lambda_fm * L_fm
        L_D = L_adv_D
    """
    def __init__(
        self,
        lambda_rec: float = 10.0,
        lambda_kl: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_fm: float = 5.0,
        lambda_perc: float = 0.1,
        adv_loss_type: str = "hinge",
        rec_loss_type: str = "l1_ssim",
        kl_free_bits: float = 0.05
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_perc = lambda_perc

        self.rec_criterion = ReconstructionLoss(loss_type=rec_loss_type)
        self.kl_criterion = KLDivergenceLoss(free_bits=kl_free_bits)
        self.adv_criterion = AdversarialLoss(loss_type=adv_loss_type)
        self.fm_criterion = FeatureMatchingLoss()
        self.perc_criterion = VGGPerceptualLoss()

    def compute_discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_rec_pred: torch.Tensor,
        fake_prior_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Discriminator must distinguish real data from both reconstructed and prior-generated samples.
        """
        # Discriminator distinguishes real vs reconstructed
        loss_d_rec = self.adv_criterion.discriminator_loss(real_pred, fake_rec_pred)
        # Discriminator distinguishes real vs prior-generated
        loss_d_prior = self.adv_criterion.discriminator_loss(real_pred, fake_prior_pred)

        total_d_loss = 0.5 * (loss_d_rec + loss_d_prior)

        metrics = {
            "loss_d_total": total_d_loss.item(),
            "loss_d_rec": loss_d_rec.item(),
            "loss_d_prior": loss_d_prior.item()
        }
        return total_d_loss, metrics

    def compute_generator_and_encoder_loss(
        self,
        x_real: torch.Tensor,
        x_rec: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        fake_rec_pred: torch.Tensor,
        fake_prior_pred: torch.Tensor,
        real_features: List[torch.Tensor],
        fake_rec_features: List[torch.Tensor],
        beta_kl: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Joint optimization of Encoder and Generator.
        """
        # 1. Pixel/Structural reconstruction
        loss_rec = self.rec_criterion(x_rec, x_real)

        # 2. Latent KL regularizer
        loss_kl = self.kl_criterion(mu, logvar)

        # 3. Adversarial fooling loss
        loss_adv_rec = self.adv_criterion.generator_loss(fake_rec_pred)
        loss_adv_prior = self.adv_criterion.generator_loss(fake_prior_pred)
        loss_adv = 0.5 * (loss_adv_rec + loss_adv_prior)

        # 4. Feature matching loss
        loss_fm = self.fm_criterion(real_features, fake_rec_features)

        # 5. Perceptual loss
        loss_perc = self.perc_criterion(x_rec, x_real)

        # Total combined loss
        total_g_loss = (
            self.lambda_rec * loss_rec
            + (self.lambda_kl * beta_kl) * loss_kl
            + self.lambda_adv * loss_adv
            + self.lambda_fm * loss_fm
            + self.lambda_perc * loss_perc
        )

        metrics = {
            "loss_g_total": total_g_loss.item(),
            "loss_rec": loss_rec.item(),
            "loss_kl": loss_kl.item(),
            "loss_adv": loss_adv.item(),
            "loss_fm": loss_fm.item(),
            "loss_perc": loss_perc.item(),
            "beta_kl": beta_kl
        }
        return total_g_loss, metrics
