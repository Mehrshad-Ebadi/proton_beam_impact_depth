import os
import time

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from unet import UNet2D
from dataset_range import OpenKBPRangeSliceDataset


def train(root: str,
          epochs: int = 40,
          bs: int = 8,
          lr: float = 1e-3,
          device: str | None = None,
          save_path: str = "range_unet_residual.pt",
          max_pts: int = 40,
          patience: int = 5,
          min_delta: float = 1e-4):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    full_ds = OpenKBPRangeSliceDataset(
        root=root,
        split="train",
        axis=0,
        voxel_mm=2.0,
        margin=5,
        max_pts=max_pts,
    )

    val_sz = max(1, int(len(full_ds) * 0.1))
    train_sz = len(full_ds) - val_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])

    train_loader = DataLoader(train_ds, batch_size=bs,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=bs,
                            shuffle=False, num_workers=0, pin_memory=False)

    model = UNet2D(in_ch=1, out_ch=1, base=64).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_ep = 0
    no_improve = 0

    for ep in range(1, epochs + 1):
        start = time.time()

        model.train()
        train_loss = 0.0

        for ct, wet_true, wet_base in train_loader:
            ct = ct.to(device)
            wet_true = wet_true.to(device)
            wet_base = wet_base.to(device)

            optimizer.zero_grad()

            residual = model(ct)
            wet_pred = wet_base + residual

            loss = criterion(wet_pred, wet_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ct.size(0)

        train_loss = train_loss / train_sz

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ct, wet_true, wet_base in val_loader:
                ct = ct.to(device)
                wet_true = wet_true.to(device)
                wet_base = wet_base.to(device)

                residual = model(ct)
                wet_pred = wet_base + residual

                loss = criterion(wet_pred, wet_true)
                val_loss += loss.item() * ct.size(0)

        val_loss = val_loss / val_sz
        elapsed = time.time() - start

        mae_cm = val_loss * 20.0

        print(f"Epoch {ep:02d} | "
              f"train={train_loss:.4f} | "
              f"val={val_loss:.4f} | "
              f"MAE≈{mae_cm:.2f}cm | "
              f"{elapsed:.1f}s")

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_ep = ep
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Stopping early at epoch {ep}. "
                      f"Best was epoch {best_ep} with val={best_loss:.4f} "
                      f"(MAE≈{best_loss * 20.0:.2f}cm)")
                break

    print(f"Done. Best val loss: {best_loss:.4f} (MAE≈{best_loss * 20.0:.2f}cm)")


if __name__ == "__main__":
    ROOT = "/content/open-kbp"
    SAVE = "/content/range_unet_residual.pt"

    train(
        root=ROOT,
        epochs=40,
        bs=8,
        lr=1e-3,
        save_path=SAVE,
        max_pts=40,
        patience=6,
        min_delta=1e-4,
    )