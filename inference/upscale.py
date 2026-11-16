import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models.super_resolution import VAESRGAN


def upscale_image(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    scale_factor: int = 4,
    device: str = "cpu"
):
    """
    Upscales an input image using VAE-SRGAN.
    """
    dev = torch.device(device)
    model = VAESRGAN(scale_factor=scale_factor).to(dev)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=dev)
        model.load_state_dict(state.get("model_state", state))
        print(f"Loaded SR checkpoint from {checkpoint_path}")
    else:
        print(f"SR Checkpoint {checkpoint_path} not found. Running inference with model structure.")

    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    lr_tensor = transform(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        sr_tensor = model.super_resolve(lr_tensor, deterministic=True)

    sr_01 = (sr_tensor.squeeze(0).cpu() + 1.0) / 2.0
    sr_01 = torch.clamp(sr_01, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(sr_01, output_path)
    print(f"Super-resolved image saved to -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to low-resolution image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/sr/sr_epoch_010.pth")
    parser.add_argument("--output", type=str, default="./outputs/sr_result.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    upscale_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device
    )
