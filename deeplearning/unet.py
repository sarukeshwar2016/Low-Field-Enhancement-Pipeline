# ============================================================
# LIGHTWEIGHT U-NET FOR MRI ENHANCEMENT
# ============================================================
import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Lightweight 2-level U-Net with skip connections.
    Input/Output: (B, 1, H, W) single-channel 2D slices.
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Pad to make dimensions divisible by 4
        _, _, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        x = nn.functional.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)

        # Remove padding
        return out[:, :, :h, :w]


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
