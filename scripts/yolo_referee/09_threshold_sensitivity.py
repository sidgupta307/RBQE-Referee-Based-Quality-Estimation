import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
    r"\09_threshold_sensitivity"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# FAILURE THRESHOLDS
# ============================================================

FAILURE_THRESHOLDS = [

    0.30,
    0.40,
    0.50,
    0.60,
    0.70

]

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

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print("=" * 70)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 70)

print(
    f"Images Loaded : {len(df)}"
)

required_columns = [

    "dice",

    "agreement_dice",
    "agreement_iou",
    "boundary_agreement",
    "area_ratio",
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

print()

print("Failure Thresholds")

for t in FAILURE_THRESHOLDS:

    print(f"Dice < {t:.2f}")

print()

print("Features")

for feature in FEATURES:

    print(feature)

results = []

print()

print("=" * 70)
print("DATA VALIDATION COMPLETE")
print("=" * 70)

# ============================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

print()
print("=" * 70)
print("RUNNING THRESHOLD SENSITIVITY")
print("=" * 70)

roc_curve_data = {}

for failure_threshold in FAILURE_THRESHOLDS:

    print()
    print(f"Failure Threshold : Dice < {failure_threshold:.2f}")

    # --------------------------------------------------------
    # Create failure labels
    # --------------------------------------------------------

    y_true = (
        df["dice"] < failure_threshold
    ).astype(int).values

    positives = int(y_true.sum())
    negatives = len(y_true) - positives

    print(
        f"Hard Cases : {positives}"
    )

    print(
        f"Easy Cases : {negatives}"
    )

    # Skip impossible thresholds

    if positives == 0 or negatives == 0:

        print(
            "Skipping (only one class present)."
        )

        continue

    roc_curve_data[
        failure_threshold
    ] = {}

    # --------------------------------------------------------
    # Evaluate every feature
    # --------------------------------------------------------

    for feature_name, feature_column in FEATURES.items():

        scores = df[
            feature_column
        ].values.astype(float)

        # Higher score must indicate
        # higher probability of failure

        if feature_column != "centroid_distance":

            scores = -scores

        # --------------------------------------------
        # ROC
        # --------------------------------------------

        auc = roc_auc_score(

            y_true,
            scores

        )

        fpr, tpr, thresholds = roc_curve(

            y_true,
            scores

        )

        roc_curve_data[
            failure_threshold
        ][
            feature_name
        ] = {

            "fpr": fpr,
            "tpr": tpr

        }

        # --------------------------------------------
        # Best threshold
        # --------------------------------------------

        best_index = np.argmax(

            tpr - fpr

        )

        best_threshold = thresholds[
            best_index
        ]

        prediction = (

            scores >= best_threshold

        ).astype(int)

        accuracy = accuracy_score(

            y_true,
            prediction

        )

        precision = precision_score(

            y_true,
            prediction,
            zero_division=0

        )

        recall = recall_score(

            y_true,
            prediction,
            zero_division=0

        )

        f1 = f1_score(

            y_true,
            prediction,
            zero_division=0

        )

        results.append({

            "Failure_Threshold":
                failure_threshold,

            "Signal":
                feature_name,

            "Feature":
                feature_column,

            "ROC_AUC":
                auc,

            "Best_Threshold":
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

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(

    by=[
        "Failure_Threshold",
        "ROC_AUC"
    ],

    ascending=[
        True,
        False
    ]

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
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(results_df)

# ============================================================
# SAVE RESULTS
# ============================================================

results_csv = os.path.join(

    OUTPUT_DIR,

    "threshold_sensitivity_results.csv"

)

results_df.to_csv(

    results_csv,

    index=False

)

# ============================================================
# HEATMAP TABLE
# ============================================================

heatmap_df = results_df.pivot(

    index="Failure_Threshold",

    columns="Signal",

    values="ROC_AUC"

)

plt.figure(figsize=(10, 5))

image = plt.imshow(

    heatmap_df.values,

    aspect="auto",

    interpolation="nearest"

)

plt.colorbar(
    image,
    label="ROC-AUC"
)

plt.xticks(

    np.arange(len(heatmap_df.columns)),

    heatmap_df.columns,

    rotation=30,

    ha="right"

)

plt.yticks(

    np.arange(len(heatmap_df.index)),

    [f"{x:.2f}" for x in heatmap_df.index]

)

plt.xlabel(
    "Agreement Signal"
)

plt.ylabel(
    "Failure Threshold (Dice)"
)

plt.title(
    "Threshold Sensitivity (ROC-AUC)"
)

plt.tight_layout()

heatmap_path = os.path.join(

    OUTPUT_DIR,

    "threshold_sensitivity_heatmap.png"

)

plt.savefig(

    heatmap_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# ROC-AUC CURVES
# ============================================================

plt.figure(figsize=(9, 6))

for feature_name in FEATURES.keys():

    subset = results_df[
        results_df["Signal"] == feature_name
    ]

    plt.plot(

        subset["Failure_Threshold"],

        subset["ROC_AUC"],

        marker="o",

        linewidth=2,

        label=feature_name

    )

plt.xlabel(
    "Failure Threshold (Dice)"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "Threshold Sensitivity Analysis"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

curve_path = os.path.join(

    OUTPUT_DIR,

    "threshold_sensitivity_curves.png"

)

plt.savefig(

    curve_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# BEST FEATURE AT EACH THRESHOLD
# ============================================================

print()

print("=" * 70)
print("BEST FEATURE PER FAILURE THRESHOLD")
print("=" * 70)

best_df = (

    results_df
    .sort_values(
        ["Failure_Threshold", "ROC_AUC"],
        ascending=[True, False]
    )
    .groupby(
        "Failure_Threshold",
        as_index=False
    )
    .first()

)

print(

    best_df[
        [
            "Failure_Threshold",
            "Signal",
            "ROC_AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    ]

)

# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 70)
print("THRESHOLD SENSITIVITY ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Images Evaluated : {len(df)}"
)

print(
    f"Failure Thresholds : {len(FAILURE_THRESHOLDS)}"
)

print(
    f"Signals Evaluated : {len(FEATURES)}"
)

print(
    f"Total Evaluations : {len(results_df)}"
)

print()

print("Results saved to:")

print(results_csv)

print(heatmap_path)

print(curve_path)