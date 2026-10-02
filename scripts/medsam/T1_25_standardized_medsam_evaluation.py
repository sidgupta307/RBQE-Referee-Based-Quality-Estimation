"""
==============================================================
T1_25_standardized_medsam_evaluation.py
==============================================================

Standardized evaluation of Independent MedSAM on the identical
1223-image benchmark used for SegFormer and UNet++.

Evaluation protocol:
- Master benchmark = clean_agreement_features.csv (1223 images)
- Independent MedSAM agreement features
- YOLO Dice (<0.50 defines failure)

Outputs
-------
failure_detection_results.csv

==============================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# INPUTS
# ============================================================

MASTER_CSV = (
    r"C:\seg_uncertain\chat_gpt\20B_segformer_clean_agreement"
    r"\clean_agreement_features.csv"
)

MEDSAM_CSV = (
    r"C:\seg_uncertain\chat_gpt\29_medsam_independent_comparison"
    r"\agreement_features_independent.csv"
)

YOLO_RESULTS = (
    r"C:\seg_uncertain\chat_gpt\01_external_validation"
    r"\combined_results.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_04_MedSAM_1223"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

master = pd.read_csv(MASTER_CSV)
medsam = pd.read_csv(MEDSAM_CSV)
results = pd.read_csv(YOLO_RESULTS)

# ============================================================
# KEEP ONLY YOLO RESULTS
# ============================================================

results = results[
    results["model"] == "YOLOv8-Seg"
].copy()

# ============================================================
# NORMALIZE IMAGE NAMES
# ============================================================

def normalize(df):

    df = df.copy()

    df["image"] = (
        df["image"]
        .str.lower()
        .str.strip()
        .str.replace(".png", "", regex=False)
        .str.replace(".jpg", "", regex=False)
    )

    return df


master = normalize(master)
medsam = normalize(medsam)
results = normalize(results)

# ============================================================
# CREATE MASTER BENCHMARK
# ============================================================

master = master[
    [
        "dataset",
        "image"
    ]
].copy()

# ============================================================
# MERGE MASTER + MEDSAM
# ============================================================

merged = master.merge(
    medsam,
    on=[
        "dataset",
        "image"
    ],
    how="inner"
)

# ============================================================
# MERGE YOLO DICE
# ============================================================

merged = merged.merge(
    results[
        [
            "dataset",
            "image",
            "dice"
        ]
    ],
    on=[
        "dataset",
        "image"
    ],
    how="inner"
)

# ============================================================
# CHECK
# ============================================================

print()

print("Master :", len(master))
print("MedSAM :", len(medsam))
print("YOLO   :", len(results))
print("Merged :", len(merged))

assert len(merged) == 1223

# ============================================================
# FAILURE LABEL
# ============================================================

FAILURE_THRESHOLD = 0.50

merged["failure"] = (
    merged["dice"] < FAILURE_THRESHOLD
).astype(int)

print()
print("Failures     :", merged["failure"].sum())
print("Non Failures :", len(merged) - merged["failure"].sum())

# ============================================================
# FEATURES
# ============================================================

FEATURES = {
    "Agreement Dice": "agreement_dice",
    "Agreement IoU": "agreement_iou",
    "Area Ratio": "area_ratio",
    "Boundary Agreement": "boundary_agreement",
    "Centroid Distance": "centroid_distance",
}

results_list = []

# ============================================================
# EVALUATION
# ============================================================

for signal_name, feature in FEATURES.items():

    df = merged[
        [feature, "failure"]
    ].dropna()

    y_true = df["failure"].values

    values = df[feature].values

    # ---------------------------------------------
    # Agreement metrics
    # Higher = better
    # Invert so larger score => failure
    # ---------------------------------------------

    if feature != "centroid_distance":

        scores = -values

    # ---------------------------------------------
    # Distance
    # Higher = worse
    # ---------------------------------------------

    else:

        scores = values

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores
    )

    best_idx = np.argmax(
        tpr - fpr
    )

    threshold = thresholds[best_idx]

    pred = (
        scores >= threshold
    ).astype(int)

    results_list.append({

        "Signal": signal_name,

        "ROC_AUC": roc_auc,

        "Threshold": threshold,

        "Accuracy":
            accuracy_score(
                y_true,
                pred
            ),

        "Precision":
            precision_score(
                y_true,
                pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                pred,
                zero_division=0
            )

    })

# ============================================================
# SAVE
# ============================================================

results_df = (
    pd.DataFrame(results_list)
    .sort_values(
        "ROC_AUC",
        ascending=False
    )
)

save_csv = os.path.join(
    OUTPUT_DIR,
    "failure_detection_results.csv"
)

results_df.to_csv(
    save_csv,
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print()
print(results_df.to_string(index=False))

print()
print("Saved to:")
print(save_csv)

print()
print("Done.")