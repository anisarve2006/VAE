import torch
import numpy as np
from typing import Tuple, List


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two vectors.
    Crucial for exploring Gaussian latent spaces where mass lies on a hypersphere.
    """
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)

    omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=-1, keepdim=True), -1.0, 1.0))
    so = torch.sin(omega)

    # Avoid division by zero when vectors are nearly collinear
    if torch.abs(so).mean() < 1e-6:
        return (1.0 - val) * low + val * high

    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def interpolate_latents(
    z1: torch.Tensor,
    z2: torch.Tensor,
    steps: int = 10,
    use_slerp: bool = True
) -> torch.Tensor:
    """
    Generates a sequence of interpolated latent vectors from z1 to z2.
    Returns: Tensor of shape [steps, latent_dim]
    """
    alphas = np.linspace(0.0, 1.0, steps)
    interpolations = []

    for alpha in alphas:
        if use_slerp:
            z_interp = slerp(float(alpha), z1, z2)
        else:
            z_interp = (1.0 - alpha) * z1 + alpha * z2
        interpolations.append(z_interp)

    return torch.cat(interpolations, dim=0)


def generate_class_transition(
    model,
    class_a: int,
    class_b: int,
    steps: int = 8,
    latent_dim: int = 128,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Generates a transition between two classes while interpolating latent codes.
    """
    model.eval()
    with torch.no_grad():
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)
        z_seq = interpolate_latents(z1, z2, steps=steps, use_slerp=True)

        # Discrete class label transition or conditioned generation
        images = []
        for i, z in enumerate(z_seq):
            label = class_a if i < (steps // 2) else class_b
            label_t = torch.tensor([label], device=device)
            if hasattr(model, "decode"):
                img = model.decode(z.unsqueeze(0), label_t)
            elif hasattr(model, "generator"):
                img = model.generator(z.unsqueeze(0), label_t)
            else:
                img = model(z.unsqueeze(0), label_t)
            images.append(img)

        return torch.cat(images, dim=0)
