import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


class PairedSRDataset(Dataset):
    """
    Dataset for Super-Resolution paired training.
    Extracts random crops from HR images and generates bicubic downsampled LR counterparts.
    """
    def __init__(self, root_dir: str, scale_factor: int = 4, crop_size: int = 128, is_train: bool = True):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.is_train = is_train
        self.image_paths = []

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        if os.path.isdir(root_dir):
            for fname in sorted(os.listdir(root_dir)):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.image_paths.append(os.path.join(root_dir, fname))

        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size) if is_train else transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return max(len(self.image_paths), 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.image_paths:
            # Fallback synthetic gradient pattern if folder is empty
            img = Image.new("RGB", (self.crop_size, self.crop_size), color=(idx * 20 % 255, 128, 200))
        else:
            img = Image.open(self.image_paths[idx % len(self.image_paths)]).convert("RGB")

        # Ensure image is large enough for crop
        if img.width < self.crop_size or img.height < self.crop_size:
            img = img.resize((max(img.width, self.crop_size), max(img.height, self.crop_size)), Image.BICUBIC)

        hr_tensor = self.hr_transform(img)

        # Generate LR tensor using bicubic downsampling
        lr_size = self.crop_size // self.scale_factor
        # Downsample tensor from [3, H, W]
        # Rescale back to PIL for authentic bicubic degradation
        to_pil = transforms.ToPILImage()
        unnorm = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        hr_denorm = unnorm(hr_tensor.clone()).clamp(0, 1)
        hr_pil = to_pil(hr_denorm)
        lr_pil = hr_pil.resize((lr_size, lr_size), Image.BICUBIC)

        lr_tensor = transforms.ToTensor()(lr_pil)
        lr_tensor = norm(lr_tensor)

        return lr_tensor, hr_tensor


class SyntheticDataset(Dataset):
    """
    Synthetic dataset producing geometric patterns and shapes for fast smoke-testing
    without downloading external datasets.
    """
    def __init__(self, num_samples: int = 500, image_size: int = 32, num_classes: int = 10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = idx % self.num_classes
        # Generate reproducible synthetic tensor based on label
        torch.manual_seed(idx)
        pattern = torch.randn(3, self.image_size, self.image_size) * 0.2
        # Add class-dependent color bias
        bias = torch.tensor([
            ((label * 3) % 10) / 10.0 - 0.5,
            ((label * 7) % 10) / 10.0 - 0.5,
            ((label * 5) % 10) / 10.0 - 0.5
        ]).view(3, 1, 1)
        pattern = (pattern + bias).clamp(-1.0, 1.0)
        return pattern, label


def get_cifar10_loaders(
    data_dir: str = "./data/raw",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads CIFAR-10 with standard [-1, 1] normalization and data augmentation.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    os.makedirs(data_dir, exist_ok=True)
    try:
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)
    except Exception as e:
        print(f"[DataLoader Warning] Could not load/download CIFAR-10 ({e}). Using SyntheticDataset fallback.")
        train_set = SyntheticDataset(num_samples=1000, image_size=32, num_classes=10)
        test_set = SyntheticDataset(num_samples=200, image_size=32, num_classes=10)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


def get_sr_loaders(
    data_dir: str = "./data/raw/sr",
    scale_factor: int = 4,
    crop_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads paired SR data from a folder or falls back gracefully.
    """
    train_dataset = PairedSRDataset(data_dir, scale_factor=scale_factor, crop_size=crop_size, is_train=True)
    val_dataset = PairedSRDataset(data_dir, scale_factor=scale_factor, crop_size=crop_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
