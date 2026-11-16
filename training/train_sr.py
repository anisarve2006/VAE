import os
import sys
import argparse
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.super_resolution import VAESRGAN
from losses.reconstruction import ReconstructionLoss
from losses.kl_divergence import KLDivergenceLoss
from losses.adversarial import AdversarialLoss
from losses.perceptual import VGGPerceptualLoss
from data.dataloaders import get_sr_loaders
from training.trainer_base import BaseTrainer


class SRTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(
            checkpoint_dir=config["training"].get("checkpoint_dir", "./checkpoints/sr"),
            output_dir=config["training"].get("output_dir", "./outputs/sr"),
            device=config.get("device", "auto")
        )
        self.config = config

        m_cfg = config["model"]
        self.model = VAESRGAN(
            in_channels=m_cfg.get("in_channels", 3),
            out_channels=m_cfg.get("out_channels", 3),
            base_channels=m_cfg.get("base_channels", 64),
            latent_dim=m_cfg.get("latent_dim", 64),
            num_rrdb=m_cfg.get("num_rrdb_blocks", 8),
            scale_factor=m_cfg.get("scale_factor", 4)
        ).to(self.device)

        t_cfg = config["training"]
        l_cfg = config.get("loss_weights", {})

        # Optimizers
        self.opt_g = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.generator.parameters()),
            lr=t_cfg.get("lr_generator", 0.0001),
            betas=(t_cfg.get("beta1", 0.9), t_cfg.get("beta2", 0.999))
        )
        self.opt_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=t_cfg.get("lr_discriminator", 0.0001),
            betas=(t_cfg.get("beta1", 0.9), t_cfg.get("beta2", 0.999))
        )

        # Loss weights
        self.lambda_pixel = l_cfg.get("lambda_pixel", 10.0)
        self.lambda_perc = l_cfg.get("lambda_perc", 1.0)
        self.lambda_adv = l_cfg.get("lambda_adv", 0.05)
        self.lambda_kl = l_cfg.get("lambda_kl", 0.01)

        self.pixel_criterion = ReconstructionLoss(loss_type="charbonnier").to(self.device)
        self.perc_criterion = VGGPerceptualLoss().to(self.device)
        self.adv_criterion = AdversarialLoss(loss_type="relativistic").to(self.device)
        self.kl_criterion = KLDivergenceLoss().to(self.device)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_d, total_g, total_pix, total_perc, total_adv, total_kl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)

            # 1. Forward Pass
            out = self.model(lr_imgs)
            sr_imgs = out["x_sr"]
            mu = out["mu"]
            logvar = out["logvar"]

            # 2. Update Discriminator
            self.opt_d.zero_grad()
            d_real = self.model.discriminator(hr_imgs)
            d_fake = self.model.discriminator(sr_imgs.detach())

            loss_d = self.adv_criterion.discriminator_loss(d_real, d_fake)
            loss_d.backward()
            self.opt_d.step()

            # 3. Update Generator & Encoder
            self.opt_g.zero_grad()
            d_fake_for_g = self.model.discriminator(sr_imgs)
            d_real_for_g = self.model.discriminator(hr_imgs)

            l_pix = self.pixel_criterion(sr_imgs, hr_imgs)
            l_perc = self.perc_criterion(sr_imgs, hr_imgs)
            l_adv = self.adv_criterion.generator_loss(d_fake_for_g, d_real_for_g)
            l_kl = self.kl_criterion(mu, logvar)

            loss_g = (
                self.lambda_pixel * l_pix
                + self.lambda_perc * l_perc
                + self.lambda_adv * l_adv
                + self.lambda_kl * l_kl
            )
            loss_g.backward()
            self.opt_g.step()

            total_d += loss_d.item()
            total_g += loss_g.item()
            total_pix += l_pix.item()
            total_perc += l_perc.item()
            total_adv += l_adv.item()
            total_kl += l_kl.item()

        n = len(dataloader)
        return {
            "loss_d": total_d / n,
            "loss_g": total_g / n,
            "loss_pix": total_pix / n,
            "loss_perc": total_perc / n,
            "loss_adv": total_adv / n,
            "loss_kl": total_kl / n
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        epochs = self.config["training"].get("epochs", 80)
        print(f"[Training VAE-SRGAN] Device: {self.device} | Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Loss_D: {metrics['loss_d']:.4f} | "
                f"Loss_G: {metrics['loss_g']:.4f} | "
                f"Pixel: {metrics['loss_pix']:.4f} | "
                f"Perc: {metrics['loss_perc']:.4f} | "
                f"Adv: {metrics['loss_adv']:.4f} | "
                f"KL: {metrics['loss_kl']:.4f}"
            )

            if epoch % self.config["training"].get("sample_interval", 5) == 0:
                self.model.eval()
                with torch.no_grad():
                    lr_sample, hr_sample = next(iter(val_loader or train_loader))
                    lr_sample = lr_sample[:4].to(self.device)
                    hr_sample = hr_sample[:4].to(self.device)

                    sr_sample = self.model.super_resolve(lr_sample)
                    # Bicubic baseline comparison
                    bicubic = F.interpolate(lr_sample, size=hr_sample.shape[-2:], mode="bicubic", align_corners=False)

                    # Interleave rows: [Bicubic, VAE-SRGAN, Ground-Truth HR]
                    combined = torch.cat([bicubic, sr_sample, hr_sample], dim=0)
                    self.save_sample_grid(combined, f"sr_comparison_epoch_{epoch:03d}.png", nrow=4)

            if epoch % self.config["training"].get("save_interval", 10) == 0:
                self.save_checkpoint({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_g": self.opt_g.state_dict(),
                    "opt_d": self.opt_d.state_dict(),
                    "config": self.config
                }, f"sr_epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sr_hybrid.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_l, val_l = get_sr_loaders(
        data_dir=cfg["data"]["data_dir"],
        scale_factor=cfg["data"]["scale_factor"],
        crop_size=cfg["data"]["crop_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    trainer = SRTrainer(cfg)
    trainer.train(train_l, val_l)
