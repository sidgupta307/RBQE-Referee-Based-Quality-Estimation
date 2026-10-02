# ============================
# 02_extract_external_tta_features.py
# PART 02A
# ============================

import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import spearmanr

from itertools import combinations

# ============================================================
# INPUT PATHS
# ============================================================

TTA_ROOT = (
    r"C:\seg_uncertain\testing"
    r"\TTA_external"
)

METRICS_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\01_external_validation"
    r"\combined_results.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\testing"
    r"\02_external_tta_features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD METRICS
# ============================================================

metrics_df = pd.read_csv(
    METRICS_CSV
)

metrics_df = metrics_df[
    metrics_df["model"] == "YOLOv8-Seg"
].copy()

print(
    f"\nLoaded images : {len(metrics_df)}"
)

# ============================================================
# DATASETS
# ============================================================

DATASETS = [

    "cvc_clinicdb",

    "cvc_colondb",

    "etis_larib",

    "cvc_300"

]

# ============================================================
# BOUNDARY FUNCTION
# ============================================================

def create_boundary_band(mask):

    mask = mask.astype(np.uint8)

    kernel = np.ones(
        (5,5),
        np.uint8
    )

    dilated = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    boundary = (
        dilated - eroded
    )

    return boundary > 0

# ============================================================
# INITIALIZE
# ============================================================

rows = []

total_images = 0

print("\nStarting feature extraction...")

# ============================================================
# FEATURE EXTRACTION
# PART 02B
# ============================================================

for dataset in DATASETS:

    print("\n" + "=" * 60)
    print(f"Processing: {dataset}")
    print("=" * 60)

    stack_dir = os.path.join(
        TTA_ROOT,
        dataset,
        "mask_stacks"
    )

    if not os.path.exists(stack_dir):

        print(f"Skipping {dataset} (missing folder)")
        continue

    stack_files = sorted([

        f for f in os.listdir(stack_dir)

        if f.endswith(".npy")

    ])

    print(
        f"Mask stacks: {len(stack_files)}"
    )

    total_images += len(stack_files)

    # --------------------------------------------
    # Metrics corresponding to current dataset
    # --------------------------------------------

    dataset_metrics = metrics_df[
        metrics_df["dataset"] == dataset
    ].copy()

    for stack_file in tqdm(stack_files):

        stack_path = os.path.join(
            stack_dir,
            stack_file
        )

        mask_stack = np.load(
            stack_path
        )

        # ====================================================
        # SANITY CHECKS
        # ====================================================

        if mask_stack.ndim != 3:

            raise RuntimeError(
                f"Invalid stack dimensions: {stack_file}"
            )

        if mask_stack.shape[0] != 9:

            raise RuntimeError(
                f"{stack_file} does not contain 9 TTA masks."
            )

        # ====================================================
        # RECOVER ORIGINAL IMAGE NAME
        # ====================================================

        stem = os.path.splitext(
            stack_file
        )[0]

        match = dataset_metrics[
            dataset_metrics["image"].str.startswith(
                stem + "."
            )
        ]

        if len(match) != 1:

            raise RuntimeError(

                f"Cannot uniquely match "

                f"{stack_file} "

                f"in dataset "

                f"{dataset}"

            )

        image_name = match.iloc[0]["image"]

        # ====================================================
        # CONSENSUS
        # ====================================================

        consensus = np.mean(
            mask_stack,
            axis=0
        )

        consensus_mask = (
            consensus > 0.5
        )

        # ====================================================
        # VARIANCE
        # ====================================================

        variance = np.var(
            mask_stack,
            axis=0
        )

        variance = variance / 0.25

        # ====================================================
        # REGIONS
        # ====================================================

        boundary_mask = create_boundary_band(
            consensus_mask
        )

        foreground_mask = (
            consensus_mask
        )

        background_mask = (
            ~consensus_mask
        )

        # ====================================================
        # GLOBAL FEATURES
        # ====================================================

        unc_mean = float(
            np.mean(variance)
        )

        unc_std = float(
            np.std(variance)
        )

        unc_max = float(
            np.max(variance)
        )

        # ====================================================
        # BOUNDARY UNCERTAINTY
        # ====================================================

        if boundary_mask.sum() > 0:

            boundary_unc = float(

                np.mean(

                    variance[
                        boundary_mask
                    ]

                )

            )

        else:

            boundary_unc = 0.0

        # ====================================================
        # FOREGROUND UNCERTAINTY
        # ====================================================

        if foreground_mask.sum() > 0:

            foreground_unc = float(

                np.mean(

                    variance[
                        foreground_mask
                    ]

                )

            )

        else:

            foreground_unc = 0.0

        # ====================================================
        # BACKGROUND UNCERTAINTY
        # ====================================================

        background_unc = float(

            np.mean(

                variance[
                    background_mask
                ]

            )

        )

        # ====================================================
        # PIXEL DISAGREEMENT
        # ====================================================

        disagreement = np.logical_and(

            consensus > 0,

            consensus < 1

        )

        pixel_disagreement_ratio = float(

            disagreement.sum()

            /

            disagreement.size

        )

        # ====================================================
        # CONSENSUS SCORE
        # ====================================================

        consensus_score = float(

            np.mean(

                np.abs(
                    consensus - 0.5
                )

            )

        )

        # ====================================================
        # CONSENSUS AREA
        # ====================================================

        consensus_area = int(
            consensus_mask.sum()
        )

        # ====================================================
        # ZERO PREDICTIONS
        # ====================================================

        num_zero_predictions = int(

            np.sum(

                [

                    mask.sum() == 0

                    for mask in mask_stack

                ]

            )

        )


        # ====================================================
        # PAIRWISE DICE AGREEMENT
        # ====================================================

        pairwise_scores = []

        for i, j in combinations(range(mask_stack.shape[0]), 2):

            m1 = mask_stack[i].astype(bool)
            m2 = mask_stack[j].astype(bool)

            area1 = m1.sum()
            area2 = m2.sum()

            if area1 == 0 and area2 == 0:
                dice = 1.0
            else:
                intersection = np.logical_and(m1, m2).sum()
                dice = (2.0 * intersection) / (area1 + area2 + 1e-8)

            pairwise_scores.append(dice)

        pairwise_scores = np.asarray(pairwise_scores)

        mean_pairwise_dice = float(pairwise_scores.mean())
        std_pairwise_dice  = float(pairwise_scores.std())
        min_pairwise_dice  = float(pairwise_scores.min())
        max_pairwise_dice  = float(pairwise_scores.max())

        # ====================================================
        # STORE FEATURES
        # ====================================================

        rows.append({

            "dataset":
                dataset,

            "image":
                image_name,

            "unc_mean":
                unc_mean,

            "unc_std":
                unc_std,

            "unc_max":
                unc_max,

            "boundary_unc":
                boundary_unc,

            "foreground_unc":
                foreground_unc,

            "background_unc":
                background_unc,

            "pixel_disagreement_ratio":
                pixel_disagreement_ratio,

            "consensus_score":
                consensus_score,

            "consensus_area":
                consensus_area,

            "mean_pairwise_dice":
                mean_pairwise_dice,

            "std_pairwise_dice":
                std_pairwise_dice,

            "min_pairwise_dice":
                min_pairwise_dice,

            "max_pairwise_dice":
                max_pairwise_dice,

            "num_zero_predictions":
                num_zero_predictions

        })

# ============================================================
# CREATE FEATURE TABLE
# PART 02C
# ============================================================

feature_df = pd.DataFrame(rows)

print("\n")
print("=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"Images processed : {len(feature_df)}")

# ============================================================
# VERIFY DUPLICATES
# ============================================================

duplicates = feature_df.duplicated(
    subset=["dataset", "image"]
).sum()

print(f"Duplicate rows : {duplicates}")

if duplicates > 0:
    raise RuntimeError(
        "Duplicate (dataset,image) pairs found."
    )

# ============================================================
# VERIFY METRICS
# ============================================================

metrics_df = metrics_df.rename(
    columns={
        "dice": "gt_dice"
    }
)

required_cols = [

    "dataset",

    "image",

    "gt_dice",

    "iou",

    "precision",

    "recall"

]

for col in required_cols:

    if col not in metrics_df.columns:

        raise RuntimeError(
            f"Missing column in metrics CSV: {col}"
        )

# ============================================================
# MERGE
# ============================================================

merged = pd.merge(

    metrics_df,

    feature_df,

    on=[

        "dataset",

        "image"

    ],

    how="inner"

)

print(f"Merged images : {len(merged)}")

# ============================================================
# VERIFY MERGE
# ============================================================

missing = len(metrics_df) - len(merged)

print(f"Images not merged : {missing}")

if missing > 0:

    print(
        "\nWARNING:"
        f" {missing} images were not matched."
    )

# ============================================================
# SAVE FEATURE TABLE
# ============================================================

FEATURE_CSV = os.path.join(

    OUTPUT_DIR,

    "tta_uncertainty_features.csv"

)

merged.to_csv(

    FEATURE_CSV,

    index=False

)

print("\nSaved feature table:")

print(FEATURE_CSV)

# ============================================================
# SPEARMAN CORRELATION ANALYSIS
# PART 02D
# ============================================================

FEATURE_COLUMNS = [

    "unc_mean",

    "unc_std",

    "unc_max",

    "boundary_unc",

    "foreground_unc",

    "background_unc",

    "pixel_disagreement_ratio",

    "consensus_score",

    "consensus_area",

    "mean_pairwise_dice",

    "std_pairwise_dice",

    "min_pairwise_dice",

    "max_pairwise_dice",

    "num_zero_predictions"

]

TARGET = "gt_dice"

corr_rows = []

print("\n")
print("=" * 60)
print("COMPUTING FEATURE CORRELATIONS")
print("=" * 60)

for feature in FEATURE_COLUMNS:

    rho, p = spearmanr(

        merged[TARGET],

        merged[feature]

    )

    corr_rows.append({

        "feature":
            feature,

        "spearman_rho":
            rho,

        "p_value":
            p,

        "abs_rho":
            abs(rho)

    })

corr_df = pd.DataFrame(
    corr_rows
)

corr_df = corr_df.sort_values(

    by="abs_rho",

    ascending=False

)

# ============================================================
# SAVE CORRELATIONS
# ============================================================

CORR_CSV = os.path.join(

    OUTPUT_DIR,

    "feature_correlations.csv"

)

corr_df.to_csv(

    CORR_CSV,

    index=False

)

# ============================================================
# SANITY CHECK
# ============================================================

print("\n")
print("=" * 60)
print("SANITY CHECK")
print("=" * 60)

print(

    merged[
        FEATURE_COLUMNS
    ].describe()

)

# ============================================================
# PRINT RANKING
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE RANKING")
print("=" * 70)

print(

    corr_df[
        [
            "feature",
            "spearman_rho",
            "p_value"
        ]
    ]

)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EXTERNAL TTA FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Images Processed : {len(merged)}"
)

print(
    f"Features Computed: {len(FEATURE_COLUMNS)}"
)

print("\nSaved Files:")

print(FEATURE_CSV)

print(CORR_CSV)