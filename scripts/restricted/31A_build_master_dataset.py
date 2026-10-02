"""
==========================================================================
31A_build_master_dataset.py
==========================================================================

Experiment:
E1 - Non-empty YOLO Prediction Control

Purpose
-------
Build a master dataset by merging:

1. SegFormer agreement features
2. YOLO Phase-4 prediction metrics

and identify images having empty YOLO predictions.

No analysis is performed in this script.

Outputs
-------
master_dataset.csv
nonempty_dataset.csv
empty_predictions.csv
dataset_summary.csv

Author : Siddharth Gupta
Project: RBQE Journal Extension
===========================================================================
"""

import os
import pandas as pd

# ==========================================================================
# INPUT FILES
# ==========================================================================

AGREEMENT_CSV = r"C:\seg_uncertain\chat_gpt\20B_segformer_clean_agreement\clean_agreement_features.csv"

YOLO_METRICS_CSV = r"C:\seg_uncertain\phase4_base_predictions\per_image_metrics.csv"

# ==========================================================================
# OUTPUT DIRECTORY
# ==========================================================================

OUTPUT_DIR = r"C:\seg_uncertain\journal_extension\31_nonempty_prediction_control\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================================
# LOAD FILES
# ==========================================================================

print("\nLoading agreement features...")
agreement_df = pd.read_csv(AGREEMENT_CSV)

print("Loading YOLO prediction metrics...")
metrics_df = pd.read_csv(YOLO_METRICS_CSV)

print("\nAgreement images :", len(agreement_df))
print("YOLO images      :", len(metrics_df))

# ==========================================================================
# KEEP ONLY REQUIRED COLUMNS
# ==========================================================================

metrics_df = metrics_df[
    [
        "image",
        "confidence",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "boundary_dice",
        "gt_area",
        "pred_area",
        "area_error",
    ]
]

# ==========================================================================
# MERGE
# ==========================================================================

master_df = agreement_df.merge(
    metrics_df,
    on="image",
    how="inner",
)

print("\nMerged images :", len(master_df))

# ==========================================================================
# CREATE EMPTY PREDICTION FLAG
# ==========================================================================

master_df["empty_prediction"] = master_df["pred_area"] == 0

master_df["prediction_status"] = master_df["empty_prediction"].map(
    {
        True: "Empty",
        False: "Non-empty",
    }
)

# ==========================================================================
# SPLIT DATA
# ==========================================================================

empty_df = master_df[
    master_df["empty_prediction"]
].copy()

nonempty_df = master_df[
    ~master_df["empty_prediction"]
].copy()

# ==========================================================================
# DATASET SUMMARY
# ==========================================================================

summary = (
    master_df
    .groupby("dataset")
    .agg(
        Total_Images=("image", "count"),
        Empty_Predictions=("empty_prediction", "sum"),
    )
)

summary["Non_Empty"] = (
    summary["Total_Images"] -
    summary["Empty_Predictions"]
)

summary["Empty_%"] = (
    100 *
    summary["Empty_Predictions"] /
    summary["Total_Images"]
)

summary = summary.reset_index()

# ==========================================================================
# SAVE FILES
# ==========================================================================

master_path = os.path.join(
    OUTPUT_DIR,
    "master_dataset.csv",
)

nonempty_path = os.path.join(
    OUTPUT_DIR,
    "nonempty_dataset.csv",
)

empty_path = os.path.join(
    OUTPUT_DIR,
    "empty_predictions.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "dataset_summary.csv",
)

master_df.to_csv(master_path, index=False)

nonempty_df.to_csv(nonempty_path, index=False)

empty_df.to_csv(empty_path, index=False)

summary.to_csv(summary_path, index=False)

# ==========================================================================
# CONSOLE SUMMARY
# ==========================================================================

print("\n" + "=" * 70)
print("MASTER DATASET SUMMARY")
print("=" * 70)

print(f"Total Images        : {len(master_df)}")
print(f"Empty Predictions   : {len(empty_df)}")
print(f"Non-empty Images    : {len(nonempty_df)}")

print("\nEmpty Prediction Images\n")

if len(empty_df):

    print(
        empty_df[
            [
                "image",
                "dataset",
                "pred_area",
                "gt_dice",
            ]
        ].to_string(index=False)
    )

print("\nDataset-wise Summary\n")

print(summary.to_string(index=False))

# ==========================================================================
# VALIDATION
# ==========================================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

assert len(master_df) == len(empty_df) + len(nonempty_df)

assert (
    master_df["image"]
    .nunique()
    ==
    len(master_df)
)

assert (
    len(nonempty_df)
    ==
    (master_df["pred_area"] > 0).sum()
)

assert (
    len(empty_df)
    ==
    (master_df["pred_area"] == 0).sum()
)

print("All validation checks passed.")

# ==========================================================================
# OUTPUT FILES
# ==========================================================================

print("\nSaved Files")

print(master_path)
print(nonempty_path)
print(empty_path)
print(summary_path)

print("\n31A COMPLETE")