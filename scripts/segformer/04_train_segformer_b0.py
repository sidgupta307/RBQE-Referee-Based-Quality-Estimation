import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

# ============================================================
# CONFIG
# ============================================================

IMG_DIR = r"C:\Polygon_polyp_new(05_26)\kvasir_dataset\images"
MASK_DIR = r"C:\Polygon_polyp_new(05_26)\kvasir_dataset\masks"

SPLIT_CSV = r"C:\seg_uncertain\phase2_yolo_training\split_manifest.csv"

OUTPUT_DIR = r"C:\seg_uncertain\segformer"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0

# TEST RUN FIRST
EPOCHS = 50

LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DATA
# ============================================================

df = pd.read_csv(SPLIT_CSV)

train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)

print("=" * 60)
print("SEGFORMER TRAINING")
print("=" * 60)
print("DEVICE:", DEVICE)
print("TRAIN :", len(train_df))
print("VAL   :", len(val_df))

# ============================================================
# PROCESSOR
# ============================================================

processor = SegformerImageProcessor(
    do_resize=False,
    do_normalize=True,
    do_rescale=True
)

# ============================================================
# DATASET
# ============================================================

class KvasirDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_name = self.df.iloc[idx]["image"]

        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        mask = (mask > 127).astype(np.uint8)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        encoded = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = encoded["pixel_values"].squeeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return pixel_values, mask

# ============================================================
# LOSSES
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):

        logits = F.interpolate(
            logits,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        targets = targets.float()

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            probs.sum() + targets.sum() + 1e-6
        )

        return 1.0 - dice

dice_loss_fn = DiceLoss()

ce_loss_fn = nn.CrossEntropyLoss()

# ============================================================
# METRIC
# ============================================================

def compute_dice(preds, targets):

    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum()

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        preds.sum() + targets.sum() + 1e-6
    )

    return dice.item()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_dataset = KvasirDataset(train_df)
    val_dataset = KvasirDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\nLoading SegFormer-B0...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    history = []

    best_dice = 0.0
    patience_counter = 0

    print("\nTraining Started...\n")

    for epoch in range(EPOCHS):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0.0

        pbar = tqdm(train_loader)

        for images, masks in pbar:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(
                pixel_values=images
            )

            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            ce_loss = ce_loss_fn(
                logits,
                masks
            )

            dice_loss = dice_loss_fn(
                outputs.logits,
                masks
            )

            loss = (
                0.5 * ce_loss +
                0.5 * dice_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch+1}/{EPOCHS} Loss {loss.item():.4f}"
            )

        train_loss /= len(train_loader)

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0
        val_dices = []

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(
                    pixel_values=images
                )

                logits = F.interpolate(
                    outputs.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                ce_loss = ce_loss_fn(
                    logits,
                    masks
                )

                dice_loss = dice_loss_fn(
                    outputs.logits,
                    masks
                )

                loss = (
                    0.5 * ce_loss +
                    0.5 * dice_loss
                )

                val_loss += loss.item()

                preds = torch.argmax(
                    logits,
                    dim=1
                )

                val_dices.append(
                    compute_dice(
                        preds,
                        masks
                    )
                )

        val_loss /= len(val_loader)

        mean_dice = float(
            np.mean(val_dices)
        )

        print(
            f"\nEpoch {epoch+1} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Dice={mean_dice:.4f}"
        )

        history.append([
            epoch + 1,
            train_loss,
            val_loss,
            mean_dice
        ])

        if mean_dice > best_dice:

            best_dice = mean_dice

            torch.save(
                model.state_dict(),
                os.path.join(
                    CHECKPOINT_DIR,
                    "best_model.pth"
                )
            )

            patience_counter = 0

            print("Best model saved.")

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping.")
            break

    # ========================================================
    # SAVE LOGS
    # ========================================================

    history_df = pd.DataFrame(
        history,
        columns=[
            "epoch",
            "train_loss",
            "val_loss",
            "val_dice"
        ]
    )

    history_df.to_csv(
        os.path.join(
            LOG_DIR,
            "training_log.csv"
        ),
        index=False
    )

    plt.figure(figsize=(8,5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train"
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation"
    )

    plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "training_curve.png"
        )
    )

    plt.close()

    print("\nTraining Complete.")
    print("Best Dice:", best_dice)