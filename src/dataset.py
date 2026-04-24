import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 2
def get_transforms(phase: str):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            # ImageNet mean/std since EfficientNet is pretrained on it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
class FakeImageDataset(Dataset):
    def __init__(self, root_dir: str, phase: str = "train"):
        self.transform = get_transforms(phase)
        self.samples: list[tuple[str, int]] = []
        label_map = {"real": 0, "fake": 1}
        for label_name, label_idx in label_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Folder not found: {folder}")
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(folder, fname), label_idx))
        print(f"[Dataset] {phase.upper()} | Total samples: {len(self.samples)}")
        self._log_distribution()
    def _log_distribution(self):
        counts = {0: 0, 1: 0}
        for _, lbl in self.samples:
            counts[lbl] += 1
        print(f" Real: {counts[0]} | Fake: {counts[1]}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
def get_dataloaders(dataset_dir: str):
    train_ds = FakeImageDataset(os.path.join(dataset_dir, "train"), phase="train")
    test_ds  = FakeImageDataset(os.path.join(dataset_dir, "test"),  phase="test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    return train_loader, test_loader
