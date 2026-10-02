import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve
)

import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\05_standardized_evaluation"
    r"\merged_1223_yolo_seed.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\08_pr_auc"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print("=" * 70)
print("PRECISION-RECALL AUC ANALYSIS")
print("=" * 70)

print(
    f"Images Loaded : {len(df)}"
)

required_columns = [

    "failure",

    "agreement_dice",
    "agreement_iou",
    "area_ratio",
    "boundary_agreement",
    "centroid_distance"

]

missing = [

    c
    for c in required_columns
    if c not in df.columns

]

if len(missing) > 0:

    raise ValueError(
        f"Missing columns: {missing}"
    )

print("All required columns found.")

print(
    f"Failures     : {df['failure'].sum()}"
)

print(
    f"Non-Failures : "
    f"{len(df) - df['failure'].sum()}"
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = {

    "Agreement Dice":
        "agreement_dice",

    "Agreement IoU":
        "agreement_iou",

    "Boundary Agreement":
        "boundary_agreement",

    "Area Ratio":
        "area_ratio",

    "Centroid Distance":
        "centroid_distance"

}

results = []

print()
print("=" * 70)
print("DATA VALIDATION COMPLETE")
print("=" * 70)

# ============================================================
# PR-AUC COMPUTATION
# ============================================================

print()
print("=" * 70)
print("PRECISION-RECALL RESULTS")
print("=" * 70)

curve_data = {}

for feature_name, feature_column in FEATURES.items():

    temp = df[
        [
            feature_column,
            "failure"
        ]
    ].dropna().reset_index(
        drop=True
    )

    y_true = temp[
        "failure"
    ].values.astype(int)

    scores = temp[
        feature_column
    ].values.astype(float)

    # --------------------------------------------------------
    # Higher score must indicate HIGHER failure probability
    # --------------------------------------------------------

    if feature_column != "centroid_distance":

        scores = -scores

    # --------------------------------------------------------
    # Precision-Recall Curve
    # --------------------------------------------------------

    precision, recall, thresholds = precision_recall_curve(

        y_true,
        scores

    )

    pr_auc = average_precision_score(

        y_true,
        scores

    )

    # --------------------------------------------------------
    # Best F1 Threshold
    # --------------------------------------------------------

    f1_scores = (

        2.0
        *
        precision[:-1]
        *
        recall[:-1]

    ) / (

        precision[:-1]
        +
        recall[:-1]
        +
        1e-12

    )

    best_index = np.argmax(
        f1_scores
    )

    best_threshold = thresholds[
        best_index
    ]

    best_precision = precision[
        best_index
    ]

    best_recall = recall[
        best_index
    ]

    best_f1 = f1_scores[
        best_index
    ]

    curve_data[
        feature_name
    ] = {

        "precision":
            precision,

        "recall":
            recall

    }

    results.append({

        "Signal":
            feature_name,

        "Feature":
            feature_column,

        "PR_AUC":
            pr_auc,

        "Best_Threshold":
            best_threshold,

        "Precision":
            best_precision,

        "Recall":
            best_recall,

        "F1":
            best_f1

    })

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(

    by="PR_AUC",

    ascending=False

).reset_index(
    drop=True
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}"
)

print()
print(results_df)

# ============================================================
# SAVE RESULTS
# ============================================================

results_csv = os.path.join(

    OUTPUT_DIR,

    "pr_auc_results.csv"

)

results_df.to_csv(

    results_csv,

    index=False

)

# ============================================================
# PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for _, row in results_df.iterrows():

    signal = row["Signal"]

    plt.plot(

        curve_data[signal]["recall"],

        curve_data[signal]["precision"],

        linewidth=2,

        label=(
            f"{signal} "
            f"(AP={row['PR_AUC']:.3f})"
        )

    )

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curves"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

figure_path = os.path.join(

    OUTPUT_DIR,

    "precision_recall_curves.png"

)

plt.savefig(

    figure_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# BEST FEATURE
# ============================================================

best = results_df.iloc[0]

print()

print("=" * 70)
print("BEST FEATURE")
print("=" * 70)

print(
    f"Signal         : {best['Signal']}"
)

print(
    f"PR-AUC         : {best['PR_AUC']:.6f}"
)

print(
    f"Best Threshold : {best['Best_Threshold']:.6f}"
)

print(
    f"Precision      : {best['Precision']:.6f}"
)

print(
    f"Recall         : {best['Recall']:.6f}"
)

print(
    f"F1 Score       : {best['F1']:.6f}"
)

# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 70)
print("PRECISION-RECALL ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Images Evaluated : {len(df)}"
)

print(
    f"Features Tested  : {len(results_df)}"
)

print()

print("Results saved to:")

print(results_csv)

print(figure_path)