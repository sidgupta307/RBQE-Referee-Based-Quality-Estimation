# ============================================================
# 03_external_tta_failure_detection.py
# PART 03A
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# INPUT
# ============================================================

FEATURE_CSV = (
    r"C:\seg_uncertain\testing"
    r"\02_external_tta_features"
    r"\tta_uncertainty_features.csv"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    r"C:\seg_uncertain\testing"
    r"\03_external_tta_failure_detection"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# FAILURE DEFINITION
# ============================================================

FAILURE_THRESHOLD = 0.50

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    FEATURE_CSV
)

print("\nLoaded images :", len(df))

# ------------------------------------------------------------
# Create binary failure label
# ------------------------------------------------------------

df["failure"] = (
    df["gt_dice"] < FAILURE_THRESHOLD
).astype(np.uint8)

print(
    "Failure cases    :",
    df["failure"].sum()
)

print(
    "Successful cases :",
    len(df) - df["failure"].sum()
)

# ============================================================
# FEATURE DEFINITIONS
#
# direction:
#   original  -> use feature directly
#   negate    -> use -feature
#
# Directions fixed using
# 03A_feature_direction_audit.py
# ============================================================

FEATURES = {

    "Zero Predictions": {

        "column":
            "num_zero_predictions",

        "direction":
            "original"

    },

    "Maximum Uncertainty": {

        "column":
            "unc_max",

        "direction":
            "negate"

    },

    "Consensus Area": {

        "column":
            "consensus_area",

        "direction":
            "negate"

    },

    "Boundary Uncertainty": {

        "column":
            "boundary_unc",

        "direction":
            "negate"

    },

    "Foreground Uncertainty": {

        "column":
            "foreground_unc",

        "direction":
            "negate"

    },

    "Std Uncertainty": {

        "column":
            "unc_std",

        "direction":
            "negate"

    },

    "Consensus Score": {

        "column":
            "consensus_score",

        "direction":
            "original"

    },

    "Mean Uncertainty": {

        "column":
            "unc_mean",

        "direction":
            "negate"

    },

    "Pixel Disagreement": {

        "column":
            "pixel_disagreement_ratio",

        "direction":
            "negate"

    },

    "Background Uncertainty": {

        "column":
            "background_unc",

        "direction":
            "negate"

    }

}

# ============================================================
# STORAGE
# ============================================================

summary_rows = []

prediction_rows = []

roc_rows = []

plt.figure(
    figsize=(8,6)
)

print("\nStarting ROC analysis...")

# ============================================================
# ROC ANALYSIS
# PART 03B
# ============================================================

for signal_name, info in FEATURES.items():

    print(f"\nProcessing: {signal_name}")

    # --------------------------------------------------------
    # Extract feature
    # --------------------------------------------------------

    feature_name = info["column"]

    direction = info["direction"]

    score = df[
        feature_name
    ].values.astype(np.float64)

    # --------------------------------------------------------
    # Apply fixed direction
    # --------------------------------------------------------

    if direction == "negate":

        score = -score

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    auc = roc_auc_score(

        df["failure"],

        score

    )

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(

        df["failure"],

        score

    )

    # --------------------------------------------------------
    # Best Threshold (Youden Index)
    # --------------------------------------------------------

    youden = tpr - fpr

    best_idx = np.argmax(
        youden
    )

    best_threshold = thresholds[
        best_idx
    ]

    # --------------------------------------------------------
    # Binary Prediction
    # --------------------------------------------------------

    pred_failure = (

        score >= best_threshold

    ).astype(np.uint8)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(

        df["failure"],

        pred_failure

    )

    precision = precision_score(

        df["failure"],

        pred_failure,

        zero_division=0

    )

    recall = recall_score(

        df["failure"],

        pred_failure,

        zero_division=0

    )

    f1 = f1_score(

        df["failure"],

        pred_failure,

        zero_division=0

    )

    # --------------------------------------------------------
    # Save Summary
    # --------------------------------------------------------

    summary_rows.append({

        "Signal":
            signal_name,

        "Feature":
            feature_name,

        "Direction":
            direction,

        "ROC_AUC":
            auc,

        "Threshold":
            best_threshold,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1

    })

    # --------------------------------------------------------
    # Save ROC Points
    # --------------------------------------------------------

    for fp, tp, th in zip(

        fpr,

        tpr,

        thresholds

    ):

        roc_rows.append({

            "Signal":
                signal_name,

            "Feature":
                feature_name,

            "Direction":
                direction,

            "Threshold":
                th,

            "FPR":
                fp,

            "TPR":
                tp

        })

    # --------------------------------------------------------
    # Save Predictions
    # --------------------------------------------------------

    pred_df = pd.DataFrame({

        "dataset":
            df["dataset"],

        "image":
            df["image"],

        "gt_dice":
            df["gt_dice"],

        "failure":
            df["failure"],

        "signal":
            signal_name,

        "feature":
            feature_name,

        "direction":
            direction,

        "score":
            score,

        "predicted_failure":
            pred_failure

    })

    prediction_rows.append(
        pred_df
    )

    # --------------------------------------------------------
    # ROC Plot
    # --------------------------------------------------------

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"{signal_name} (AUC={auc:.3f})"

    )

# ============================================================
# SAVE RESULTS
# PART 03C
# ============================================================

# ------------------------------------------------------------
# Summary Table
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    summary_rows
)

summary_df = summary_df.sort_values(
    by="ROC_AUC",
    ascending=False
).reset_index(drop=True)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "failure_detection_results.csv"
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)

# ------------------------------------------------------------
# Best Thresholds
# ------------------------------------------------------------

THRESHOLD_CSV = os.path.join(
    OUTPUT_DIR,
    "best_thresholds.csv"
)

summary_df[
    [
        "Signal",
        "Feature",
        "Direction",
        "Threshold"
    ]
].to_csv(
    THRESHOLD_CSV,
    index=False
)

# ------------------------------------------------------------
# ROC Points
# ------------------------------------------------------------

roc_df = pd.DataFrame(
    roc_rows
)

ROC_DATA_CSV = os.path.join(
    OUTPUT_DIR,
    "roc_data.csv"
)

roc_df.to_csv(
    ROC_DATA_CSV,
    index=False
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

prediction_df = pd.concat(

    prediction_rows,

    ignore_index=True

)

PREDICTION_CSV = os.path.join(

    OUTPUT_DIR,

    "predictions.csv"

)

prediction_df.to_csv(

    PREDICTION_CSV,

    index=False

)

# ------------------------------------------------------------
# ROC Figure
# ------------------------------------------------------------

plt.plot(

    [0, 1],

    [0, 1],

    "--",

    color="black",

    linewidth=1,

    label="Random"

)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "External TTA Failure Detection"
)

plt.legend(
    loc="lower right",
    fontsize=9
)

plt.grid(
    alpha=0.30
)

plt.tight_layout()

ROC_FIGURE = os.path.join(

    OUTPUT_DIR,

    "roc_curves.png"

)

plt.savefig(

    ROC_FIGURE,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ------------------------------------------------------------
# Print Results
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("FAILURE DETECTION RESULTS")
print("=" * 75)

print(

    summary_df[
        [
            "Signal",
            "Direction",
            "ROC_AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    ]

)

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(SUMMARY_CSV)
print(THRESHOLD_CSV)
print(ROC_DATA_CSV)
print(PREDICTION_CSV)
print(ROC_FIGURE)

# ============================================================
# FINAL SANITY CHECK
# PART 03D
# ============================================================

print("\n")
print("=" * 75)
print("FINAL SANITY CHECK")
print("=" * 75)

# ------------------------------------------------------------
# Check for NaNs
# ------------------------------------------------------------

print("\nNaN Values")

print(summary_df.isna().sum())

# ------------------------------------------------------------
# Verify ROC range
# ------------------------------------------------------------

if ((summary_df["ROC_AUC"] < 0).any() or
    (summary_df["ROC_AUC"] > 1).any()):

    raise RuntimeError(
        "Invalid ROC AUC detected."
    )

print("\nROC AUC values verified.")

# ------------------------------------------------------------
# Verify thresholds
# ------------------------------------------------------------

if summary_df["Threshold"].isna().any():

    raise RuntimeError(
        "Missing threshold values."
    )

print("Thresholds verified.")

# ------------------------------------------------------------
# Best Feature
# ------------------------------------------------------------

best = summary_df.iloc[0]

print("\n")
print("=" * 75)
print("BEST FEATURE")
print("=" * 75)

print(f"Signal      : {best['Signal']}")
print(f"Feature     : {best['Feature']}")
print(f"Direction   : {best['Direction']}")
print(f"ROC AUC     : {best['ROC_AUC']:.4f}")
print(f"Accuracy    : {best['Accuracy']:.4f}")
print(f"Precision   : {best['Precision']:.4f}")
print(f"Recall      : {best['Recall']:.4f}")
print(f"F1 Score    : {best['F1']:.4f}")
print(f"Threshold   : {best['Threshold']:.6f}")

# ------------------------------------------------------------
# Worst Feature
# ------------------------------------------------------------

worst = summary_df.iloc[-1]

print("\n")
print("=" * 75)
print("WORST FEATURE")
print("=" * 75)

print(f"Signal      : {worst['Signal']}")
print(f"Feature     : {worst['Feature']}")
print(f"Direction   : {worst['Direction']}")
print(f"ROC AUC     : {worst['ROC_AUC']:.4f}")

# ------------------------------------------------------------
# Dataset Summary
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("DATASET SUMMARY")
print("=" * 75)

dataset_summary = (

    df.groupby("dataset")
      .agg(

        Images=("image", "count"),

        Failures=("failure", "sum"),

        Mean_Dice=("gt_dice", "mean")

      )

)

print(dataset_summary)

# ------------------------------------------------------------
# Overall Statistics
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("OVERALL SUMMARY")
print("=" * 75)

print(f"Total Images      : {len(df)}")
print(f"Failure Images    : {df['failure'].sum()}")
print(f"Successful Images : {len(df)-df['failure'].sum()}")

print("\nMean Dice : {:.4f}".format(
    df["gt_dice"].mean()
))

print("Failure Threshold : {:.2f}".format(
    FAILURE_THRESHOLD
))

# ------------------------------------------------------------
# Save Dataset Summary
# ------------------------------------------------------------

DATASET_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "dataset_summary.csv"
)

dataset_summary.to_csv(
    DATASET_SUMMARY_CSV
)

# ------------------------------------------------------------
# Output Files
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(SUMMARY_CSV)
print(THRESHOLD_CSV)
print(ROC_DATA_CSV)
print(PREDICTION_CSV)
print(ROC_FIGURE)
print(DATASET_SUMMARY_CSV)

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("EXTERNAL TTA FAILURE DETECTION COMPLETE")
print("=" * 75)