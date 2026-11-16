import os
import torch
from torchvision.utils import make_grid, save_image


class BaseTrainer:
    """
    Abstract base trainer providing logging, checkpointing, and device management.
    """
    def __init__(
        self,
        checkpoint_dir: str,
        output_dir: str,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, state: dict, filename: str):
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        print(f"[Checkpoint Saved] -> {filepath}")

    def load_checkpoint(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        state = torch.load(filepath, map_location=self.device)
        print(f"[Checkpoint Loaded] <- {filepath}")
        return state

    def save_sample_grid(self, images: torch.Tensor, filename: str, nrow: int = 8):
        """
        Saves a grid of generated/reconstructed images scaled to [0, 1].
        """
        filepath = os.path.join(self.output_dir, filename)
        # Denormalize [-1, 1] to [0, 1]
        imgs_01 = (images.clone().detach().cpu() + 1.0) / 2.0
        imgs_01 = torch.clamp(imgs_01, 0.0, 1.0)
        grid = make_grid(imgs_01, nrow=nrow, padding=2)
        save_image(grid, filepath)
        print(f"[Sample Saved] -> {filepath}")
