import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.gan import ConditionalGAN
from losses.adversarial import AdversarialLoss
from data.dataloaders import get_cifar10_loaders
from training.trainer_base import BaseTrainer


class GANTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(
            checkpoint_dir=config["training"].get("checkpoint_dir", "./checkpoints/gan"),
            output_dir=config["training"].get("output_dir", "./outputs/gan"),
            device=config.get("device", "auto")
        )
        self.config = config

        m_cfg = config["model"]
        self.model = ConditionalGAN(
            in_channels=config["data"].get("channels", 3),
            latent_dim=m_cfg.get("latent_dim", 128),
            base_channels=m_cfg.get("base_channels", 64),
            num_classes=config["data"].get("num_classes", 10),
            condition_dim=m_cfg.get("condition_dim", 64),
            image_size=config["data"].get("image_size", 32)
        ).to(self.device)

        t_cfg = config["training"]
        self.opt_g = optim.Adam(
            self.model.generator.parameters(),
            lr=t_cfg.get("lr_generator", 0.0002),
            betas=(t_cfg.get("beta1", 0.0), t_cfg.get("beta2", 0.99))
        )
        self.opt_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=t_cfg.get("lr_discriminator", 0.0002),
            betas=(t_cfg.get("beta1", 0.0), t_cfg.get("beta2", 0.99))
        )

        self.adv_loss = AdversarialLoss(loss_type=t_cfg.get("adv_loss_type", "hinge"))

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_d_loss, total_g_loss = 0.0, 0.0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            B = images.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            self.opt_d.zero_grad()
            z = torch.randn(B, self.model.latent_dim, device=self.device)
            with torch.no_grad():
                fake_imgs = self.model.generator(z, labels)

            d_real, _ = self.model.discriminator(images, labels)
            d_fake, _ = self.model.discriminator(fake_imgs.detach(), labels)

            loss_d = self.adv_loss.discriminator_loss(d_real, d_fake)
            loss_d.backward()
            self.opt_d.step()

            # -----------------
            # Train Generator
            # -----------------
            self.opt_g.zero_grad()
            z_new = torch.randn(B, self.model.latent_dim, device=self.device)
            gen_imgs = self.model.generator(z_new, labels)
            d_gen, _ = self.model.discriminator(gen_imgs, labels)

            loss_g = self.adv_loss.generator_loss(d_gen)
            loss_g.backward()
            self.opt_g.step()

            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()

        n = len(dataloader)
        return {
            "loss_d": total_d_loss / n,
            "loss_g": total_g_loss / n
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        epochs = self.config["training"].get("epochs", 50)
        print(f"[Training cGAN] Device: {self.device} | Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss D: {metrics['loss_d']:.4f} | Loss G: {metrics['loss_g']:.4f}")

            if epoch % self.config["training"].get("sample_interval", 5) == 0:
                self.model.eval()
                with torch.no_grad():
                    fixed_labels = torch.arange(10, device=self.device).repeat(2)
                    samples = self.model.sample(len(fixed_labels), fixed_labels, device=self.device)
                    self.save_sample_grid(samples, f"cgan_samples_epoch_{epoch:03d}.png", nrow=10)

            if epoch % self.config["training"].get("save_interval", 10) == 0:
                self.save_checkpoint({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_g": self.opt_g.state_dict(),
                    "opt_d": self.opt_d.state_dict()
                }, f"cgan_epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/gan_cifar10.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_l, test_l = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    trainer = GANTrainer(cfg)
    trainer.train(train_l, test_l)
