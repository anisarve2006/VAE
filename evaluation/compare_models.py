import os
import torch
import numpy as np
from typing import Dict, List, Optional
from torchvision.utils import make_grid, save_image
from .metrics import calculate_psnr, calculate_ssim, PerceptualMetricLPIPS


def benchmark_reconstruction(
    model,
    dataloader,
    device: torch.device = torch.device("cpu"),
    num_batches: int = 5
) -> Dict[str, float]:
    """
    Evaluates reconstruction fidelity on validation data.
    """
    model.eval()
    psnr_scores = []
    ssim_scores = []
    lpips_eval = PerceptualMetricLPIPS(device=device)
    lpips_scores = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            if hasattr(model, "reconstruct"):
                recons = model.reconstruct(images, labels)
            elif hasattr(model, "forward"):
                out = model(images, labels)
                recons = out[0] if isinstance(out, (tuple, list)) else out.get("x_rec", images)
            else:
                recons = images

            # Rescale to [0, 1] for metrics
            img_01 = (images + 1.0) / 2.0
            rec_01 = (recons + 1.0) / 2.0

            psnr_scores.append(calculate_psnr(img_01, rec_01))
            ssim_scores.append(calculate_ssim(img_01, rec_01))
            lpips_scores.append(lpips_eval(img_01, rec_01))

    return {
        "PSNR (dB)": float(np.mean(psnr_scores)),
        "SSIM": float(np.mean(ssim_scores)),
        "LPIPS": float(np.mean(lpips_scores))
    }


def format_benchmark_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Formats multi-model benchmark results into a clean Markdown table.
    """
    metrics = list(next(iter(results.values())).keys())
    header = "| Model Architecture | " + " | ".join(metrics) + " |"
    divider = "| :--- | " + " | ".join([":---:" for _ in metrics]) + " |"

    rows = []
    for model_name, score_dict in results.items():
        row_str = f"| **{model_name}** | " + " | ".join([f"{score_dict[m]:.4f}" for m in metrics]) + " |"
        rows.append(row_str)

    return "\n".join([header, divider] + rows)
