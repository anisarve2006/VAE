import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.conditional_vae_gan import ConditionalVAEGAN
from losses.hybrid_loss import HybridVAEGANLoss
from losses.kl_divergence import AnnealingScheduler
from data.dataloaders import get_cifar10_loaders
from training.trainer_base import BaseTrainer


class HybridTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(
            checkpoint_dir=config["training"].get("checkpoint_dir", "./checkpoints/hybrid"),
            output_dir=config["training"].get("output_dir", "./outputs/hybrid"),
            device=config.get("device", "auto")
        )
        self.config = config

        m_cfg = config["model"]
        self.model = ConditionalVAEGAN(
            in_channels=config["data"].get("channels", 3),
            latent_dim=m_cfg.get("latent_dim", 128),
            base_channels=m_cfg.get("base_channels", 64),
            num_classes=config["data"].get("num_classes", 10),
            condition_dim=m_cfg.get("condition_dim", 64),
            image_size=config["data"].get("image_size", 32)
        ).to(self.device)

        t_cfg = config["training"]
        l_cfg = config.get("loss_weights", {})

        # Optimizers: Separate for (Encoder + Generator) vs Discriminator
        self.opt_eg = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.generator.parameters()),
            lr=t_cfg.get("lr_generator", 0.0002),
            betas=(t_cfg.get("beta1", 0.5), t_cfg.get("beta2", 0.999))
        )
        self.opt_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=t_cfg.get("lr_discriminator", 0.0001),
            betas=(t_cfg.get("beta1", 0.5), t_cfg.get("beta2", 0.999))
        )

        # Unified Loss
        self.criterion = HybridVAEGANLoss(
            lambda_rec=l_cfg.get("lambda_rec", 10.0),
            lambda_kl=l_cfg.get("lambda_kl", 1.0),
            lambda_adv=l_cfg.get("lambda_adv", 1.0),
            lambda_fm=l_cfg.get("lambda_fm", 5.0),
            lambda_perc=l_cfg.get("lambda_perc", 0.1),
            adv_loss_type=t_cfg.get("adv_loss_type", "hinge"),
            kl_free_bits=t_cfg.get("kl_free_bits", 0.05)
        )

        self.scheduler = AnnealingScheduler(
            total_epochs=t_cfg.get("epochs", 60),
            warmup_epochs=t_cfg.get("kl_anneal_epochs", 10),
            max_beta=1.0
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        accum_metrics = {}
        beta = self.scheduler.get_beta(epoch)

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            B = images.size(0)

            # ----------------------------------------------------
            # 1. Forward Pass through Hybrid Architecture
            # ----------------------------------------------------
            forward_out = self.model(images, labels)
            x_rec = forward_out["x_rec"]
            x_prior = forward_out["x_prior"]
            mu = forward_out["mu"]
            logvar = forward_out["logvar"]

            # ----------------------------------------------------
            # 2. Update Discriminator
            # ----------------------------------------------------
            self.opt_d.zero_grad()
            d_real, _ = self.model.discriminator(images, labels)
            d_rec_fake, _ = self.model.discriminator(x_rec.detach(), labels)
            d_prior_fake, _ = self.model.discriminator(x_prior.detach(), labels)

            loss_d, d_metrics = self.criterion.compute_discriminator_loss(
                real_pred=d_real,
                fake_rec_pred=d_rec_fake,
                fake_prior_pred=d_prior_fake
            )
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.config["training"].get("grad_clip_norm", 5.0))
            self.opt_d.step()

            # ----------------------------------------------------
            # 3. Update Encoder & Generator Jointly
            # ----------------------------------------------------
            self.opt_eg.zero_grad()
            # Extract features for feature matching and adversarial gradients
            d_real_for_fm, real_feats = self.model.discriminator(images, labels, return_features=True)
            d_rec_for_g, rec_feats = self.model.discriminator(x_rec, labels, return_features=True)
            d_prior_for_g, _ = self.model.discriminator(x_prior, labels, return_features=False)

            loss_eg, eg_metrics = self.criterion.compute_generator_and_encoder_loss(
                x_real=images,
                x_rec=x_rec,
                mu=mu,
                logvar=logvar,
                fake_rec_pred=d_rec_for_g,
                fake_prior_pred=d_prior_for_g,
                real_features=real_feats,
                fake_rec_features=rec_feats,
                beta_kl=beta
            )
            loss_eg.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) + list(self.model.generator.parameters()),
                self.config["training"].get("grad_clip_norm", 5.0)
            )
            self.opt_eg.step()

            # Merge step metrics
            for k, v in {**d_metrics, **eg_metrics}.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v

        n = len(dataloader)
        return {k: v / n for k, v in accum_metrics.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        epochs = self.config["training"].get("epochs", 60)
        print(f"[Training VAE-GAN Hybrid] Device: {self.device} | Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Loss_D: {metrics['loss_d_total']:.4f} | "
                f"Loss_G: {metrics['loss_g_total']:.4f} | "
                f"Rec: {metrics['loss_rec']:.4f} | "
                f"KL: {metrics['loss_kl']:.4f} | "
                f"FM: {metrics['loss_fm']:.4f} | "
                f"Perc: {metrics['loss_perc']:.4f} | "
                f"Beta: {metrics['beta_kl']:.2f}"
            )

            if epoch % self.config["training"].get("sample_interval", 5) == 0:
                self.model.eval()
                with torch.no_grad():
                    # Generate visual comparison: [Original, Reconstruction, Prior Sample]
                    fixed_images, fixed_labels = next(iter(val_loader or train_loader))
                    fixed_images = fixed_images[:8].to(self.device)
                    fixed_labels = fixed_labels[:8].to(self.device)

                    recons = self.model.reconstruct(fixed_images, fixed_labels)
                    priors = self.model.sample(8, fixed_labels, device=self.device)

                    # Interleave [real, recon, prior] rows
                    combined = torch.cat([fixed_images, recons, priors], dim=0)
                    self.save_sample_grid(combined, f"hybrid_comparison_epoch_{epoch:03d}.png", nrow=8)

            if epoch % self.config["training"].get("save_interval", 10) == 0:
                self.save_checkpoint({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_eg": self.opt_eg.state_dict(),
                    "opt_d": self.opt_d.state_dict(),
                    "config": self.config
                }, f"hybrid_epoch_{epoch:03d}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/vae_gan_cifar10.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_l, test_l = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    trainer = HybridTrainer(cfg)
    trainer.train(train_l, test_l)
