#!/usr/bin/env python3
"""
Reviewer 6, Comment 5 - controlled cross-architecture referee.

Retrains the UNet++ (ResNet34) referee with EXACTLY the SegFormer-B0 training
protocol (scripts/segformer/04_train_segformer_b0.py), so that the two
cross-architecture referees differ only in architecture:

  same Kvasir-SEG split      split_manifest.csv (700 train / 200 val)
  same input size            512 x 512 (train and inference)
  same preprocessing         RGB, /255, ImageNet mean/std, no augmentation
  same optimizer             AdamW, lr 1e-4, weight decay 1e-4
  same batch / epochs        batch 8, up to 50 epochs, patience 10
  same loss                  0.5 * cross-entropy-type loss + 0.5 * soft Dice
                             (binary form for a one-channel output)
  same model selection       checkpoint with the best validation Dice

Usage (Windows, from C:\\seg_uncertain):
  python R5B_train_unetpp_matched.py --gpu 1
Output:
  C:\\seg_uncertain\\unetpp_matched\\checkpoints\\best_unetpp_matched.pth
  C:\\seg_uncertain\\unetpp_matched\\logs\\training_log.csv
"""
import argparse
import os
import sys

_gpu = "1"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
    elif arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

p = argparse.ArgumentParser()
p.add_argument("--gpu", default="1")
p.add_argument("--img_dir", default=r"C:\Polygon_polyp_new(05_26)\kvasir_dataset\images")
p.add_argument("--mask_dir", default=r"C:\Polygon_polyp_new(05_26)\kvasir_dataset\masks")
p.add_argument("--split_csv", default=r"C:\seg_uncertain\phase2_yolo_training\split_manifest.csv")
p.add_argument("--out_dir", default=r"C:\seg_uncertain\unetpp_matched")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
DEVICE = "cuda"
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")

# ---- protocol identical to SegFormer-B0 ----------------------------------
IMAGE_SIZE, BATCH_SIZE, EPOCHS = 512, 8, 50
LR, WEIGHT_DECAY, PATIENCE = 1e-4, 1e-4, 10
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

ckpt_dir = os.path.join(a.out_dir, "checkpoints"); log_dir = os.path.join(a.out_dir, "logs")
os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

df = pd.read_csv(a.split_csv)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)
print("TRAIN:", len(train_df), " VAL:", len(val_df))


class KvasirDataset(Dataset):
    def __init__(self, frame):
        self.df = frame

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        name = self.df.iloc[i]["image"]
        img = cv2.cvtColor(cv2.imread(os.path.join(a.img_dir, name)), cv2.COLOR_BGR2RGB)
        msk = (cv2.imread(os.path.join(a.mask_dir, name), 0) > 127).astype(np.uint8)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(msk).float().unsqueeze(0)


def soft_dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + 1e-6) / (prob.sum() + target.sum() + 1e-6)


bce = nn.BCEWithLogitsLoss()
g = torch.Generator(); g.manual_seed(a.seed)
train_loader = DataLoader(KvasirDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
val_loader = DataLoader(KvasirDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation=None).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best, wait, hist = 0.0, 0, []
for ep in range(1, EPOCHS + 1):
    model.train(); tl = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = 0.5 * bce(out, y) + 0.5 * soft_dice_loss(out, y)
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
    tl /= len(train_loader)

    model.eval(); vl, vd = 0.0, []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            vl += (0.5 * bce(out, y) + 0.5 * soft_dice_loss(out, y)).item()
            pred = (torch.sigmoid(out) > 0.5).float()
            vd.append(((2 * (pred * y).sum() + 1e-6) / (pred.sum() + y.sum() + 1e-6)).item())
    vl /= len(val_loader); md = float(np.mean(vd))
    hist.append((ep, tl, vl, md))
    print(f"Epoch {ep} | train loss {tl:.4f} | val loss {vl:.4f} | val Dice {md:.4f}")
    if md > best:
        best, wait = md, 0
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_unetpp_matched.pth"))
        print("  best model saved")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping."); break

pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss", "val_dice"]).to_csv(
    os.path.join(log_dir, "training_log.csv"), index=False)
print(f"\nBest validation Dice = {best:.4f}")
print("Weights:", os.path.join(ckpt_dir, "best_unetpp_matched.pth"))
