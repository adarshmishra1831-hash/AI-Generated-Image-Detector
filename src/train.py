import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from dataset import get_dataloaders
from model import AIImageDetector

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
OUTPUT_DIR   = "outputs"
EPOCHS       = 3
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ── Plot Training History ─────────────────────────────────────────────────────
def plot_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="tomato")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", color="royalblue")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="tomato")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Training history saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  AI Image Detector — Training")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"{'='*55}\n")

    train_loader, val_loader = get_dataloaders(DATASET_DIR)

    model     = AIImageDetector(num_classes=2, dropout=0.4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history    = {"train_loss": [], "val_loss": [],
                  "train_acc":  [], "val_acc":  []}
    best_acc   = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch [{epoch}/{EPOCHS}]  LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Train  → Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"  Val    → Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%")
        print(f"  Time   → {elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc   = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  ✅ Best model saved (Val Acc: {best_acc*100:.2f}%)")

    print(f"\n🏁 Training complete. Best Val Acc: {best_acc*100:.2f}% "
          f"at Epoch {best_epoch}")

    # Save final model + history plot
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    plot_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    # Save history to CSV for reference
    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)


if __name__ == "__main__":
    main()
