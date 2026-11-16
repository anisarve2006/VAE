import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 Multi-layer Perceptual Loss.
    Measures semantic feature distance rather than pixel-wise discrepancies.
    """
    def __init__(self, feature_layers=(2, 7, 12, 21, 30), weights=(1.0, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights

        try:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
            self.vgg_available = True
        except Exception as e:
            print(f"[Warning] Could not download VGG-19 weights ({e}). Initializing random VGG architecture.")
            vgg = vgg19(weights=None).features
            self.vgg_available = False

        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in feature_layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, layer_idx + 1)]))
            prev = layer_idx + 1

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Scale from [-1, 1] to [0, 1]
        x_01 = (x + 1.0) / 2.0
        return (x_01 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)

        loss = 0.0
        h_pred = pred_norm
        h_target = target_norm

        for slice_layer, w in zip(self.slices, self.weights):
            h_pred = slice_layer(h_pred)
            with torch.no_grad():
                h_target = slice_layer(h_target)
            loss += w * nn.functional.l1_loss(h_pred, h_target)

        return loss
