"""
Train BrainTumorCNN on real, prepared BraTS 2021 FLAIR slices
(output of prepare_brats.py).

Run order:
  1. python3 prepare_brats.py     # converts raw BraTS .nii.gz -> brats_prepared/
  2. python3 train_real.py        # trains on brats_prepared/, saves model_weights.pt

This produces the SAME model_weights.pt format that train_demo.py does, so
the FastAPI backend (app/main.py) and the React frontend work identically
either way — only the data source changes.

Implements report Ch. 6.2's augmentation step (rotation + horizontal flip,
applied ONLY to the training split) and reports accuracy, precision,
recall, F1 and a confusion matrix, matching a typical Phase-2 results
chapter.
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from app.model import BrainTumorCNN

IMG_SIZE = 128
DATA_DIR = "brats_prepared"
MODEL_OUT = "model_weights.pt"
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3


class BraTSSliceDataset(Dataset):
    def __init__(self, samples, augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, img):
        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        # Random small rotation (+/- 15 deg)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            m = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
            img = cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.augment:
            img = self._augment(img)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor, torch.tensor([label], dtype=torch.float32)


def collect_samples(root):
    samples = []
    for label, cls in enumerate(["notumor", "tumor"]):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            samples.append((os.path.join(cls_dir, f), label))
    return samples


def train():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"'{DATA_DIR}' not found. Run prepare_brats.py first.")

    samples = collect_samples(DATA_DIR)
    labels = [s[1] for s in samples]
    print(f"Total slices: {len(samples)}  (tumor={sum(labels)}, notumor={len(labels)-sum(labels)})")

    train_samples, temp_samples = train_test_split(
        samples, test_size=0.3, stratify=labels, random_state=42
    )
    temp_labels = [s[1] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, stratify=temp_labels, random_state=42
    )
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    train_ds = BraTSSliceDataset(train_samples, augment=True)
    val_ds = BraTSSliceDataset(val_samples, augment=False)
    test_ds = BraTSSliceDataset(test_samples, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = BrainTumorCNN(in_channels=1, input_size=IMG_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = (torch.sigmoid(model(x)) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={total_loss/len(train_ds):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)

    # ---- final test-set evaluation ----
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = (torch.sigmoid(model(x)) > 0.5).float().cpu().numpy()
            all_preds.extend(pred.flatten().tolist())
            all_labels.extend(y.numpy().flatten().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== Test set results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(f"\nBest model saved to {MODEL_OUT}")


if __name__ == "__main__":
    train()
