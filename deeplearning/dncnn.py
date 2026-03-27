# ============================================================
# DnCNN — DENOISING CONVOLUTIONAL NEURAL NETWORK
# ============================================================
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Residual-learning DnCNN.
    Learns the noise residual: output = input - noise_estimate.
    Input/Output: (B, 1, H, W) single-channel 2D slices.
    """

    def __init__(self, depth=7, channels=64):
        super().__init__()

        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(1, channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (no activation — outputs noise estimate)
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return x - noise  # Residual learning


if __name__ == "__main__":
    model = DnCNN()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
