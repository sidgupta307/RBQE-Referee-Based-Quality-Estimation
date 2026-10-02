"""
==========================================================================
31B_nonempty_control_auc.py
==========================================================================

Experiment
----------
E1 - Non-Empty YOLO Prediction Control

Reviewer Request
----------------
Evaluate whether the reported Agreement Dice ROC-AUC remains stable after
removing images where the initial YOLO prediction is empty or nearly empty.

Method
------
1. Load the complete Agreement Dice feature table.
2. Load the corresponding YOLO prediction masks.
3. Compute the predicted foreground area ratio for every image.
4. Remove images with foreground area < 1% of the image.
5. Recompute Agreement Dice ROC-AUC on the filtered dataset.
6. Compare against the full dataset.

Inputs
------
C:\\seg_uncertain\\chat_gpt\\20B_segformer_clean_agreement\\clean_agreement_features.csv

C:\\seg_uncertain\\chat_gpt\\02_external_predictions

Outputs
-------
agreement_full_dataset.csv
agreement_nonempty_dataset.csv
removed_images.csv
summary.csv
summary.txt

Author : Siddharth Gupta
Project : RBQE Journal Extension
==========================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

# ============================================================
# INPUT FILES
# ============================================================

AGREEMENT_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\20B_segformer_clean_agreement"
    r"\clean_agreement_features.csv"
)

YOLO_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\02_external_predictions"
)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    r"C:\seg_uncertain"
    r"\journal_extension"
    r"\31_nonempty_prediction_control"
    r"\outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# PARAMETERS
# ============================================================

FOREGROUND_THRESHOLD = 0.01      # 1% of image area

FAILURE_THRESHOLD = 0.50         # Dice < 0.50

# ============================================================
# LOAD AGREEMENT DATASET
# ============================================================

print("\n" + "=" * 90)
print("NON-EMPTY YOLO PREDICTION CONTROL")
print("=" * 90)

print("\nLoading agreement feature table...")

agreement_df = pd.read_csv(
    AGREEMENT_CSV
)

print(f"Images Loaded : {len(agreement_df)}")

required_columns = [
    "dataset",
    "image",
    "gt_dice",
    "agreement_dice"
]

missing = [
    c for c in required_columns
    if c not in agreement_df.columns
]

if len(missing):

    raise ValueError(
        f"Missing columns : {missing}"
    )

print("\nAgreement Dataset")
print("------------------------------")
print(agreement_df.head())

print("\nDataset Distribution")
print("------------------------------")
print(
    agreement_df["dataset"]
    .value_counts()
    .sort_index()
)

# ============================================================
# COMPUTE YOLO FOREGROUND AREA
# ============================================================

print("\n" + "=" * 90)
print("MEASURING YOLO PREDICTED FOREGROUND AREA")
print("=" * 90)

foreground_pixels = []
foreground_ratio = []
mask_paths = []

missing_masks = 0

for _, row in agreement_df.iterrows():

    dataset = row["dataset"]

    image_name = row["image"]

    mask_name = os.path.splitext(image_name)[0] + ".png"

    mask_path = os.path.join(
        YOLO_ROOT,
        dataset,
        "masks",
        mask_name
    )

    mask_paths.append(mask_path)

    if not os.path.exists(mask_path):

        foreground_pixels.append(np.nan)
        foreground_ratio.append(np.nan)

        missing_masks += 1

        continue

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        foreground_pixels.append(np.nan)
        foreground_ratio.append(np.nan)

        missing_masks += 1

        continue

    binary_mask = mask > 0

    fg_pixels = int(binary_mask.sum())

    total_pixels = binary_mask.size

    ratio = fg_pixels / total_pixels

    foreground_pixels.append(fg_pixels)

    foreground_ratio.append(ratio)

agreement_df["mask_path"] = mask_paths
agreement_df["foreground_pixels"] = foreground_pixels
agreement_df["foreground_ratio"] = foreground_ratio

print("\nMask Statistics")
print("-" * 40)

print(f"Missing Masks : {missing_masks}")
print(f"Valid Masks   : {len(agreement_df) - missing_masks}")

if missing_masks > 0:

    raise RuntimeError(
        f"{missing_masks} prediction masks could not be found."
    )

print("\nForeground Area Ratio Summary")
print("-" * 40)

print(
    agreement_df["foreground_ratio"]
    .describe()
)

print("\nSmallest Predictions")
print("-" * 40)

print(
    agreement_df[
        [
            "dataset",
            "image",
            "foreground_pixels",
            "foreground_ratio"
        ]
    ]
    .sort_values(
        "foreground_ratio"
    )
    .head(20)
    .to_string(index=False)
)

# ============================================================
# CREATE NON-EMPTY SUBSET
# ============================================================

print("\n" + "=" * 90)
print("CREATING NON-EMPTY SUBSET")
print("=" * 90)

full_df = agreement_df.copy()

nonempty_df = agreement_df[
    agreement_df["foreground_pixels"] > 0
].copy()

removed_df = agreement_df[
    agreement_df["foreground_pixels"] == 0
].copy()

print(f"\nTotal Images     : {len(full_df)}")
print(f"Removed Images   : {len(removed_df)}")
print(f"Remaining Images : {len(nonempty_df)}")

print(
    f"Removal %        : "
    f"{100*len(removed_df)/len(full_df):.2f}%"
)

# ============================================================
# ROC-AUC
# ============================================================

print("\n" + "=" * 90)
print("ROC ANALYSIS")
print("=" * 90)

full_df["hard_case"] = (
    full_df["gt_dice"] < FAILURE_THRESHOLD
).astype(int)

nonempty_df["hard_case"] = (
    nonempty_df["gt_dice"] < FAILURE_THRESHOLD
).astype(int)

full_scores = (
    1.0 -
    full_df["agreement_dice"]
).values

nonempty_scores = (
    1.0 -
    nonempty_df["agreement_dice"]
).values

full_auc = roc_auc_score(
    full_df["hard_case"],
    full_scores
)

nonempty_auc = roc_auc_score(
    nonempty_df["hard_case"],
    nonempty_scores
)

print(f"\nFull Dataset ROC-AUC      : {full_auc:.6f}")
print(f"Non-empty Dataset ROC-AUC : {nonempty_auc:.6f}")
print(f"AUC Difference            : {nonempty_auc-full_auc:.6f}")

# ============================================================
# SAVE DATASETS
# ============================================================

full_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "agreement_full_dataset.csv"
    ),
    index=False
)

nonempty_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "agreement_nonempty_dataset.csv"
    ),
    index=False
)

removed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "removed_images.csv"
    ),
    index=False
)

# ============================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 90)

thresholds = [
    0.0000,
    0.0005,
    0.0010,
    0.0025,
    0.0050,
    0.0100,
]

summary_rows = []

for thr in thresholds:

    subset = agreement_df[
        agreement_df["foreground_ratio"] >= thr
    ].copy()

    subset["hard_case"] = (
        subset["gt_dice"] < FAILURE_THRESHOLD
    ).astype(int)

    failure_score = (
        1.0 -
        subset["agreement_dice"]
    ).values

    auc = roc_auc_score(
        subset["hard_case"],
        failure_score
    )

    summary_rows.append({

        "Foreground_Threshold": thr,

        "Images_Remaining": len(subset),

        "Images_Removed":
            len(agreement_df) - len(subset),

        "Removal_Percentage":
            100.0 *
            (len(agreement_df)-len(subset))
            / len(agreement_df),

        "ROC_AUC": auc,

        "Delta_AUC":
            auc - full_auc

    })

summary_df = pd.DataFrame(summary_rows)

print("\nThreshold Sensitivity")
print("-"*90)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:f"{x:.6f}"
    )
)

# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary_csv = os.path.join(
    OUTPUT_DIR,
    "summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

# ============================================================
# SAVE TEXT REPORT
# ============================================================

summary_txt = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)

with open(summary_txt,"w") as f:

    f.write("="*90 + "\n")
    f.write("NON-EMPTY YOLO PREDICTION CONTROL\n")
    f.write("="*90 + "\n\n")

    f.write(f"Total Images            : {len(full_df)}\n")
    f.write(f"Removed Images (1%)     : {len(removed_df)}\n")
    f.write(f"Remaining Images        : {len(nonempty_df)}\n")
    f.write(f"Removal Percentage      : {100*len(removed_df)/len(full_df):.2f}%\n\n")

    f.write(f"Full Dataset ROC-AUC    : {full_auc:.6f}\n")
    f.write(f"Non-empty ROC-AUC       : {nonempty_auc:.6f}\n")
    f.write(f"Delta ROC-AUC           : {nonempty_auc-full_auc:.6f}\n\n")

    f.write("Threshold Sensitivity\n")
    f.write("-"*90 + "\n")

    f.write(
        summary_df.to_string(
            index=False,
            float_format=lambda x:f"{x:.6f}"
        )
    )

print("\n")
print("="*90)
print("FILES SAVED")
print("="*90)

print(
    os.path.join(
        OUTPUT_DIR,
        "agreement_full_dataset.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "agreement_nonempty_dataset.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "removed_images.csv"
    )
)

print(summary_csv)
print(summary_txt)

print("\n")
print("="*90)
print("NON-EMPTY CONTROL COMPLETE")
print("="*90)

# ============================================================
# DIAGNOSTIC ANALYSIS OF REMOVED IMAGES
# ============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC ANALYSIS OF REMOVED IMAGES")
print("=" * 90)

removed_df["hard_case"] = (
    removed_df["gt_dice"] < FAILURE_THRESHOLD
).astype(int)

print(f"\nRemoved Images : {len(removed_df)}")

print("\nGround Truth Dice Statistics")
print("-" * 40)

print(
    removed_df["gt_dice"].describe()
)

print("\nAgreement Dice Statistics")
print("-" * 40)

print(
    removed_df["agreement_dice"].describe()
)

num_hard = int(removed_df["hard_case"].sum())
num_easy = len(removed_df) - num_hard

print("\nFailure Distribution")
print("-" * 40)

print(f"Hard Cases (Dice < 0.50) : {num_hard}")
print(f"Easy Cases               : {num_easy}")

print(
    f"Hard Case Percentage     : "
    f"{100*num_hard/len(removed_df):.2f}%"
)

print("\nDataset Distribution")
print("-" * 40)

print(
    removed_df["dataset"]
    .value_counts()
    .sort_index()
)

diagnostic_csv = os.path.join(
    OUTPUT_DIR,
    "removed_images_diagnostic.csv"
)

removed_df.to_csv(
    diagnostic_csv,
    index=False
)

print("\nDiagnostic CSV Saved")

print(diagnostic_csv)