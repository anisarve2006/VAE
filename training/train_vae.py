import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vae import ConditionalVAE
from losses.reconstruction import ReconstructionLoss
from losses.kl_divergence import KLDivergenceLoss, AnnealingScheduler
from data.dataloaders import get_cifar10_loaders
from training.trainer_base import BaseTrainer


class VAETrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(
            checkpoint_dir=config["training"].get("checkpoint_dir", "./checkpoints/vae"),
            output_dir=config["training"].get("output_dir", "./outputs/vae"),
            device=config.get("device", "auto")
        )
        self.config = config

        # Model
        m_cfg = config["model"]
        self.model = ConditionalVAE(
            in_channels=config["data"].get("channels", 3),
            latent_dim=m_cfg.get("latent_dim", 128),
            base_channels=m_cfg.get("base_channels", 64),
            num_classes=config["data"].get("num_classes", 10),
            condition_dim=m_cfg.get("condition_dim", 64),
            image_size=config["data"].get("image_size", 32)
        ).to(self.device)

        # Optimizers
        t_cfg = config["training"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=t_cfg.get("lr_encoder", 0.0002),
            betas=(t_cfg.get("beta1", 0.5), t_cfg.get("beta2", 0.999)),
            weight_decay=float(t_cfg.get("weight_decay", 1e-5))
        )

        # Losses
        self.rec_loss = ReconstructionLoss(loss_type=t_cfg.get("rec_loss_type", "l1_ssim"))
        self.kl_loss = KLDivergenceLoss(free_bits=t_cfg.get("kl_free_bits", 0.1))
        self.scheduler = AnnealingScheduler(
            total_epochs=t_cfg.get("epochs", 50),
            warmup_epochs=t_cfg.get("kl_anneal_epochs", 10),
            max_beta=t_cfg.get("kl_weight", 1.0)
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss, total_rec, total_kl = 0.0, 0.0, 0.0
        beta = self.scheduler.get_beta(epoch)

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            recon_x, mu, logvar, z = self.model(images, labels)

            l_rec = self.rec_loss(recon_x, images)
            l_kl = self.kl_loss(mu, logvar)
            loss = l_rec + beta * l_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"].get("grad_clip_norm", 5.0))
            self.optimizer.step()

            total_loss += loss.item()
            total_rec += l_rec.item()
            total_kl += l_kl.item()

        n_batches = len(dataloader)
        return {
            "loss": total_loss / n_batches,
            "rec_loss": total_rec / n_batches,
            "kl_loss": total_kl / n_batches,
            "beta": beta
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        epochs = self.config["training"].get("epochs", 50)
        print(f"[Training CVAE] Device: {self.device} | Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {metrics['loss']:.4f} | Rec: {metrics['rec_loss']:.4f} | KL: {metrics['kl_loss']:.4f} | Beta: {metrics['beta']:.2f}")

            if epoch % self.config["training"].get("sample_interval", 5) == 0:
                self.model.eval()
                with torch.no_grad():
                    fixed_labels = torch.arange(10, device=self.device).repeat(2) # 20 samples
                    samples = self.model.sample(len(fixed_labels), fixed_labels, device=self.device)
                    self.save_sample_grid(samples, f"cvae_samples_epoch_{epoch:03d}.png", nrow=10)

            if epoch % self.config["training"].get("save_interval", 10) == 0:
                self.save_checkpoint({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "config": self.config
                }, f"cvae_epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/vae_cifar10.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_l, test_l = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    trainer = VAETrainer(cfg)
    trainer.train(train_l, test_l)
