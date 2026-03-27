# ============================================================
# TRAINING SCRIPT — U-NET + DnCNN (CPU)
# ============================================================
import os
import sys
import time
import torch
import torch.optim as optim

# Add parent for imports
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import get_dataloaders
from unet import UNet
from dncnn import DnCNN

SAVE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 4


def train_model(model, name, train_loader, epochs=EPOCHS):
    """Train a single model and save weights."""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for lf, hf, _, _ in train_loader:
            lf, hf = lf.to(DEVICE), hf.to(DEVICE)

            pred = model(lf)
            loss = loss_fn(pred, hf)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss={avg_loss:.6f} | Time={elapsed:.1f}s")

    # Save model
    save_path = os.path.join(SAVE_DIR, f"{name.lower()}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  ✔ Saved → {save_path}")
    return model


def main():
    print("Loading dataset...")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Train U-Net
    unet = UNet()
    train_model(unet, "UNet", train_loader)

    # Train DnCNN
    dncnn = DnCNN()
    train_model(dncnn, "DnCNN", train_loader)

    print("\n✔ All models trained and saved.")


if __name__ == "__main__":
    main()
