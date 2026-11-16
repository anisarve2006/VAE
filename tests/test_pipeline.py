import unittest
import torch

from models.encoder import ConditionalEncoder
from models.decoder import ConditionalDecoder
from models.discriminator import ProjectionDiscriminator
from models.conditional_vae_gan import ConditionalVAEGAN
from models.super_resolution import VAESRGAN
from losses.hybrid_loss import HybridVAEGANLoss
from losses.kl_divergence import KLDivergenceLoss
from evaluation.latent_analysis import slerp, interpolate_latents
from evaluation.metrics import calculate_psnr, calculate_ssim


class TestVAEGANPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.channels = 3
        self.size = 32
        self.latent_dim = 64
        self.num_classes = 10

        self.dummy_imgs = torch.randn(self.batch_size, self.channels, self.size, self.size).clamp(-1.0, 1.0)
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_encoder_shapes_and_grad(self):
        encoder = ConditionalEncoder(
            in_channels=self.channels,
            latent_dim=self.latent_dim,
            base_channels=16,
            num_classes=self.num_classes,
            image_size=self.size
        )
        mu, logvar = encoder(self.dummy_imgs, self.dummy_labels)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

        z = encoder.reparameterize(mu, logvar)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # Check gradient flow
        loss = mu.sum() + logvar.sum()
        loss.backward()
        self.assertIsNotNone(encoder.in_conv.weight.grad)

    def test_decoder_shapes(self):
        decoder = ConditionalDecoder(
            latent_dim=self.latent_dim,
            out_channels=self.channels,
            base_channels=16,
            num_classes=self.num_classes,
            target_size=self.size
        )
        z = torch.randn(self.batch_size, self.latent_dim)
        out = decoder(z, self.dummy_labels)
        self.assertEqual(out.shape, (self.batch_size, self.channels, self.size, self.size))
        self.assertTrue((out >= -1.0).all() and (out <= 1.0).all())

    def test_discriminator_and_features(self):
        d = ProjectionDiscriminator(
            in_channels=self.channels,
            base_channels=16,
            num_classes=self.num_classes,
            image_size=self.size
        )
        logits, features = d(self.dummy_imgs, self.dummy_labels, return_features=True)
        self.assertEqual(logits.shape, (self.batch_size, 1))
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_hybrid_forward_and_loss(self):
        model = ConditionalVAEGAN(
            in_channels=self.channels,
            latent_dim=self.latent_dim,
            base_channels=16,
            num_classes=self.num_classes,
            image_size=self.size
        )
        res = model(self.dummy_imgs, self.dummy_labels)
        self.assertIn("x_rec", res)
        self.assertIn("x_prior", res)
        self.assertEqual(res["x_rec"].shape, self.dummy_imgs.shape)

        criterion = HybridVAEGANLoss(lambda_rec=1.0, lambda_kl=1.0, lambda_adv=1.0, lambda_fm=1.0, lambda_perc=0.0)

        # Test loss computation
        d_real, real_feats = model.discriminator(self.dummy_imgs, self.dummy_labels, return_features=True)
        d_rec, rec_feats = model.discriminator(res["x_rec"], self.dummy_labels, return_features=True)
        d_prior, _ = model.discriminator(res["x_prior"], self.dummy_labels, return_features=False)

        loss_d, d_metrics = criterion.compute_discriminator_loss(d_real, d_rec.detach(), d_prior.detach())
        loss_eg, eg_metrics = criterion.compute_generator_and_encoder_loss(
            self.dummy_imgs, res["x_rec"], res["mu"], res["logvar"],
            d_rec, d_prior, real_feats, rec_feats
        )

        self.assertFalse(torch.isnan(loss_d))
        self.assertFalse(torch.isnan(loss_eg))

    def test_super_resolution_vae_gan(self):
        sr_model = VAESRGAN(in_channels=3, out_channels=3, base_channels=16, latent_dim=32, num_rrdb=2, scale_factor=4)
        lr_input = torch.randn(2, 3, 16, 16).clamp(-1.0, 1.0)
        out = sr_model(lr_input)
        self.assertEqual(out["x_sr"].shape, (2, 3, 64, 64))

        inference_sr = sr_model.super_resolve(lr_input)
        self.assertEqual(inference_sr.shape, (2, 3, 64, 64))

    def test_latent_interpolation(self):
        z1 = torch.randn(1, self.latent_dim)
        z2 = torch.randn(1, self.latent_dim)
        z_seq = interpolate_latents(z1, z2, steps=6, use_slerp=True)
        self.assertEqual(z_seq.shape, (6, self.latent_dim))

    def test_metrics(self):
        img1 = torch.rand(2, 3, 32, 32)
        img2 = img1.clone()
        psnr = calculate_psnr(img1, img2)
        ssim = calculate_ssim(img1, img2)
        self.assertEqual(psnr, float("inf"))
        self.assertAlmostEqual(ssim, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
