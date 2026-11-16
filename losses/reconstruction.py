import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Smooth L1 loss variant: sqrt((x - y)^2 + eps^2).
    Offers sharper reconstruction than L2 while maintaining continuous derivatives at zero.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class DifferentiableSSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) Loss.
    Maximizing SSIM is equivalent to minimizing 1 - SSIM.
    """
    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        # Create 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        # Create 2D Gaussian kernel
        _2d_window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", _2d_window.expand(channels, 1, window_size, window_size).contiguous())

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Scale [-1, 1] to [0, 1] for SSIM computation
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        padd = self.window_size // 2
        window = self.window.to(img1.device, dtype=img1.dtype)
        mu1 = F.conv2d(img1, window, padding=padd, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=padd, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=self.channels) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class ReconstructionLoss(nn.Module):
    """
    Combined Pixel + Structural Reconstruction Loss.
    """
    def __init__(self, loss_type: str = "l1_ssim", ssim_weight: float = 0.2):
        super().__init__()
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        self.ssim = DifferentiableSSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return self.l1(pred, target)
        elif self.loss_type == "l2":
            return self.l2(pred, target)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(pred, target)
        elif self.loss_type == "l1_ssim":
            return self.l1(pred, target) + self.ssim_weight * self.ssim(pred, target)
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.loss_type}")
