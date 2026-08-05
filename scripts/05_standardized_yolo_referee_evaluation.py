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

# ============================================================
# PATHS
# ============================================================

INPUT_CSV = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\04_yolo_seed_agreement"
    r"\merged_1223_yolo_seed.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\05_standardized_evaluation"
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
print("YOLO SEED REFEREE EVALUATION")
print("=" * 70)

print(f"Images Loaded : {len(df)}")

required_columns = [

    "dataset",
    "image",
    "dice",
    "failure",

    "agreement_dice",
    "agreement_iou",
    "area_ratio",
    "boundary_agreement",
    "centroid_distance"

]

missing = [

    c for c in required_columns
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
    f"{len(df)-df['failure'].sum()}"
)

# ============================================================
# FEATURES TO EVALUATE
# ============================================================

FEATURES = {

    "Agreement Dice":
        "agreement_dice",

    "Agreement IoU":
        "agreement_iou",

    "Area Ratio":
        "area_ratio",

    "Boundary Agreement":
        "boundary_agreement",

    "Centroid Distance":
        "centroid_distance"

}

results = []

# ============================================================
# EVALUATE SINGLE FEATURE
# ============================================================

def evaluate_feature(

    dataframe,
    feature_name,
    feature_column

):

    temp = dataframe[
        [
            feature_column,
            "failure"
        ]
    ].dropna()

    y_true = temp[
        "failure"
    ].values.astype(int)

    scores = temp[
        feature_column
    ].values.astype(float)

    # --------------------------------------------------------
    # Higher score must always indicate HIGHER failure risk
    # --------------------------------------------------------

    if feature_column != "centroid_distance":

        scores = -scores

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    auc = roc_auc_score(

        y_true,
        scores

    )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    fpr, tpr, thresholds = roc_curve(

        y_true,
        scores

    )

    # --------------------------------------------------------
    # YOUDEN INDEX
    # --------------------------------------------------------

    youden = tpr - fpr

    best_index = np.argmax(
        youden
    )

    best_threshold = thresholds[
        best_index
    ]

    # --------------------------------------------------------
    # FINAL PREDICTIONS
    # --------------------------------------------------------

    predictions = (
        scores >= best_threshold
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_true,
        predictions

    )

    precision = precision_score(

        y_true,
        predictions,
        zero_division=0

    )

    recall = recall_score(

        y_true,
        predictions,
        zero_division=0

    )

    f1 = f1_score(

        y_true,
        predictions,
        zero_division=0

    )

    return {

        "Signal":
            feature_name,

        "Feature":
            feature_column,

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

    }

# ============================================================
# EVALUATE ALL FEATURES
# ============================================================

print()
print("=" * 70)
print("FAILURE DETECTION RESULTS")
print("=" * 70)

for feature_name, feature_column in FEATURES.items():

    result = evaluate_feature(

        dataframe=df,

        feature_name=feature_name,

        feature_column=feature_column

    )

    results.append(
        result
    )

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(

    by="ROC_AUC",

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

    "failure_detection_results.csv"

)

results_df.to_csv(

    results_csv,

    index=False

)

benchmark_csv = os.path.join(

    OUTPUT_DIR,

    "merged_1223_yolo_seed.csv"

)

df.to_csv(

    benchmark_csv,

    index=False

)

# ============================================================
# SUMMARY
# ============================================================

best_row = results_df.iloc[0]

print()
print("=" * 70)
print("BEST FEATURE")
print("=" * 70)

print(
    f"Signal      : {best_row['Signal']}"
)

print(
    f"ROC-AUC     : {best_row['ROC_AUC']:.6f}"
)

print(
    f"Threshold   : {best_row['Threshold']:.6f}"
)

print(
    f"Accuracy    : {best_row['Accuracy']:.6f}"
)

print(
    f"Precision   : {best_row['Precision']:.6f}"
)

print(
    f"Recall      : {best_row['Recall']:.6f}"
)

print(
    f"F1 Score    : {best_row['F1']:.6f}"
)

print()
print("=" * 70)
print("YOLO SEED REFEREE EVALUATION COMPLETE")
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

print(benchmark_csv)