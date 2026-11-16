import os
import argparse
import torch
from torchvision.utils import save_image, make_grid
from models.conditional_vae_gan import ConditionalVAEGAN
from evaluation.latent_analysis import interpolate_latents, slerp


def generate_conditional_samples(
    checkpoint_path: str,
    output_path: str = "./outputs/samples.png",
    num_samples_per_class: int = 4,
    num_classes: int = 10,
    latent_dim: int = 128,
    truncation_psi: float = 0.8,
    device: str = "cpu"
):
    """
    Generates class-conditional sample grid with latent truncation.
    """
    dev = torch.device(device)
    model = ConditionalVAEGAN(latent_dim=latent_dim, num_classes=num_classes).to(dev)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=dev)
        model.load_state_dict(state.get("model_state", state))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Running with initial weights for demonstration.")

    model.eval()
    all_images = []

    with torch.no_grad():
        for c in range(num_classes):
            labels = torch.full((num_samples_per_class,), c, dtype=torch.long, device=dev)
            # Latent sampling with truncation trick
            z = torch.randn(num_samples_per_class, latent_dim, device=dev) * truncation_psi
            imgs = model.decode(z, labels)
            all_images.append(imgs)

    batch_imgs = torch.cat(all_images, dim=0)
    batch_01 = (batch_imgs.cpu() + 1.0) / 2.0
    grid = make_grid(batch_01, nrow=num_samples_per_class, padding=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(grid, output_path)
    print(f"Generated samples successfully written to -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/hybrid/hybrid_epoch_010.pth")
    parser.add_argument("--output", type=str, default="./outputs/generated_grid.png")
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate_conditional_samples(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples_per_class=args.samples_per_class,
        device=args.device
    )
