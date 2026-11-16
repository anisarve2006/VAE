import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import vgg16, VGG16_Weights, inception_v3, Inception_V3_Weights


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Accepts tensors in [0, 1] range.
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Computes structural similarity index (SSIM) between two batches of images.
    Accepts tensors in [0, 1] range.
    """
    channels = img1.size(1)
    sigma = 1.5
    gauss = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    _2d_window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channels, 1, window_size, window_size).to(img1.device)

    padd = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padd, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channels) - mu1_mu2

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


class PerceptualMetricLPIPS(nn.Module):
    """
    Self-contained LPIPS perceptual similarity calculator using pretrained VGG-16.
    Computes normalized deep cosine / L2 feature distance across Conv layers.
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device)
            self.available = True
        except Exception:
            vgg = vgg16(weights=None).features.to(device)
            self.available = False

        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg[i] for i in range(0, 4)]),   # relu1_2
            nn.Sequential(*[vgg[i] for i in range(4, 9)]),   # relu2_2
            nn.Sequential(*[vgg[i] for i in range(9, 16)]),  # relu3_3
            nn.Sequential(*[vgg[i] for i in range(16, 23)])  # relu4_3
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculates LPIPS distance (lower is more perceptually similar).
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        dist = 0.0
        h1, h2 = img1, img2
        for s in self.slices:
            h1 = s(h1)
            h2 = s(h2)
            # Normalize activations across channels
            norm1 = h1 / (torch.sqrt(torch.sum(h1 ** 2, dim=1, keepdim=True)) + 1e-10)
            norm2 = h2 / (torch.sqrt(torch.sum(h2 ** 2, dim=1, keepdim=True)) + 1e-10)
            dist += torch.mean((norm1 - norm2) ** 2).item()

        return float(dist / len(self.slices))


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calculates the Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2).
    FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
