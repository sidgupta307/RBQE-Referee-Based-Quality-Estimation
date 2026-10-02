"""
==============================================================================
48_paired_delong_referee_comparison.py
==============================================================================

Purpose
-------
Paired DeLong statistical comparison between referee-based quality estimation
methods.

Comparisons
-----------
1. SegFormer-B0 vs TTA
2. UNet++ vs TTA
3. SegFormer-B0 vs UNet++

Failure Definition
------------------
Hard Case:
    Ground Truth Dice < 0.50

Quality Signal
--------------
Agreement Dice

Outputs
-------
paired_delong_results.csv
paired_auc_summary.csv
paired_delong_summary.txt

Author : Siddharth Gupta
Project: RBQE Journal Extension
==============================================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from scipy import stats

# =============================================================================
# INPUT FILES
# =============================================================================

SEGFORMER_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\20B_segformer_clean_agreement"
    r"\clean_agreement_features.csv"
)

YOLO_REFEREE_CSV = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\05_standardized_evaluation"
    r"\merged_1223_yolo_seed.csv"
)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = (
    r"C:\seg_uncertain"
    r"\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\07_delong"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =============================================================================
# PARAMETERS
# =============================================================================

FAILURE_THRESHOLD = 0.50

# =============================================================================
# CONSOLE HEADER
# =============================================================================

print("\n")
print("=" * 90)
print("SEGFORMER vs YOLO REFEREE DELONG TEST")
print("=" * 90)

print("\nInput Files")
print("-" * 40)

print("SegFormer")
print(SEGFORMER_CSV)

print("\nYOLO Referee")
print(YOLO_REFEREE_CSV)

print("\nOutput Folder")
print(OUTPUT_DIR)

# =============================================================================
# DELONG IMPLEMENTATION
# (Same implementation used in T1_20_external_delong_test.py)
# =============================================================================

def compute_midrank(x):

    J = np.argsort(x)

    Z = x[J]

    N = len(x)

    T = np.zeros(N, dtype=float)

    i = 0

    while i < N:

        j = i

        while j < N and Z[j] == Z[i]:
            j += 1

        T[i:j] = 0.5 * (i + j - 1)

        i = j

    T2 = np.empty(N, dtype=float)

    T2[J] = T + 1

    return T2


def fast_delong(predictions_sorted_transposed,
                label_1_count):

    m = label_1_count

    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]

    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])

    ty = np.empty([k, n])

    tz = np.empty([k, m + n])

    for r in range(k):

        tx[r, :] = compute_midrank(
            positive_examples[r, :]
        )

        ty[r, :] = compute_midrank(
            negative_examples[r, :]
        )

        tz[r, :] = compute_midrank(
            predictions_sorted_transposed[r, :]
        )

    aucs = (
        tz[:, :m].sum(axis=1)
        - m * (m + 1) / 2
    ) / (m * n)

    v01 = (
        tz[:, :m] - tx
    ) / n

    v10 = (
        1.0 - (tz[:, m:] - ty) / m
    )

    sx = np.cov(v01)

    sy = np.cov(v10)

    delongcov = (
        sx / m +
        sy / n
    )

    return aucs, delongcov


def calc_pvalue(aucs,
                sigma):

    l = np.array([[1, -1]])

    z = (
        np.abs(np.diff(aucs))
        /
        np.sqrt(
            np.dot(
                np.dot(l, sigma),
                l.T
            )
        )
    )

    return (
        np.log10(2)
        +
        stats.norm.logsf(
            z,
            loc=0,
            scale=1
        ) / np.log(10)
    )


def compute_ground_truth_statistics(
        ground_truth):

    order = (-ground_truth).argsort()

    label_1_count = int(
        ground_truth.sum()
    )

    return order, label_1_count


def delong_roc_test(
        ground_truth,
        predictions_one,
        predictions_two):

    order, label_1_count = (
        compute_ground_truth_statistics(
            ground_truth
        )
    )

    predictions_sorted = np.vstack(
        (
            predictions_one,
            predictions_two,
        )
    )[:, order]

    aucs, covariance = fast_delong(
        predictions_sorted,
        label_1_count,
    )

    return calc_pvalue(
        aucs,
        covariance,
    )

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n")
print("=" * 90)
print("LOADING DATA")
print("=" * 90)

# =============================================================================
# LOAD DATA
# =============================================================================

seg_df = pd.read_csv(SEGFORMER_CSV)

yolo_df = pd.read_csv(YOLO_REFEREE_CSV)

print("\nImages Loaded")
print("-" * 40)

print(f"SegFormer    : {len(seg_df)}")
print(f"YOLO Referee : {len(yolo_df)}")

# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

required_seg = [

    "dataset",
    "image",
    "gt_dice",
    "agreement_dice"

]

required_yolo = [

    "dataset",
    "image",
    "agreement_dice"

]

for col in required_seg:

    if col not in seg_df.columns:
        raise ValueError(f"SegFormer missing column: {col}")

for col in required_yolo:

    if col not in yolo_df.columns:
        raise ValueError(f"YOLO Referee missing column: {col}")
    
print("\nColumn validation passed.")

# =============================================================================
# NORMALIZE KEYS
# =============================================================================

for dataframe in [seg_df, yolo_df]:

    dataframe["dataset"] = (
        dataframe["dataset"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    dataframe["image"] = (
        dataframe["image"]
        .astype(str)
        .str.lower()
        .str.strip()
        .apply(lambda x: os.path.splitext(x)[0])
    )

# =============================================================================
# KEEP REQUIRED COLUMNS
# =============================================================================

seg_df = seg_df[
    [
        "dataset",
        "image",
        "gt_dice",
        "agreement_dice"
    ]
].rename(
    columns={
        "agreement_dice":"segformer_score"
    }
)

yolo_df = yolo_df[
    [
        "dataset",
        "image",
        "agreement_dice"
    ]
].rename(
    columns={
        "agreement_dice":"yolo_score"
    }
)

# =============================================================================
# MERGE
# =============================================================================

merged_df = seg_df.merge(

    yolo_df,

    on=[
        "dataset",
        "image"
    ],

    how="inner"

)

print(f"\nMerged Images : {len(merged_df)}")

print("\nDataset Distribution")
print("-" * 40)

print("\nSegFormer")
print(seg_df["dataset"].value_counts())

print("\nYOLO Referee")
print(yolo_df["dataset"].value_counts())

print("\nMerged")
print(merged_df["dataset"].value_counts())

print("\nUnique datasets")
print(sorted(seg_df["dataset"].unique()))
print(sorted(yolo_df["dataset"].unique()))

# =============================================================================
# CREATE FAILURE LABEL
# =============================================================================

merged_df["failure"] = (
    merged_df["gt_dice"] < FAILURE_THRESHOLD
).astype(int)

print("\nFailure Statistics")
print("-" * 40)

print(
    "Hard Cases :",
    int(merged_df["failure"].sum())
)

print(
    "Easy Cases :",
    len(merged_df) -
    int(merged_df["failure"].sum())
)

# =============================================================================
# DATA PREVIEW
# =============================================================================

print("\n")
print("=" * 90)
print("DATA PREVIEW")
print("=" * 90)

print(

    merged_df[
        [
            "dataset",
            "image",
            "gt_dice",
            "segformer_score",
            "yolo_score",
            "failure",
        ]
    ]
    .head(10)
    .to_string(index=False)

)

# =============================================================================
# VALIDATION
# =============================================================================

#assert len(merged_df) > 1000
assert len(merged_df) > 0

assert merged_df[
    [
        "segformer_score",
        "yolo_score",
    ]
].isna().sum().sum() == 0

assert (
    merged_df[["dataset", "image"]]
    .drop_duplicates()
    .shape[0]
    == len(merged_df)
), "Duplicate (dataset, image) pairs found."

print("\nValidation Passed.")
print("-" * 40)

print(f"Images : {len(merged_df)}")
print("No missing scores.")
print("Unique (dataset, image) pairs confirmed.")

# =============================================================================
# ROC-AUC CALCULATION
# =============================================================================

print("\n")
print("=" * 90)
print("ROC-AUC CALCULATION")
print("=" * 90)

y_true = merged_df["failure"].values

segformer_scores = merged_df["segformer_score"].values
yolo_scores = merged_df["yolo_score"].values

segformer_failure_score = 1.0 - segformer_scores
yolo_failure_score = 1.0 - yolo_scores

segformer_auc = roc_auc_score(
    y_true,
    segformer_failure_score
)

yolo_auc = roc_auc_score(
    y_true,
    yolo_failure_score
)

print("\nROC-AUC Results")
print("-" * 40)

print(f"SegFormer      : {segformer_auc:.6f}")
print(f"YOLO Referee   : {yolo_auc:.6f}")


# =============================================================================
# PAIRED DELONG TESTS
# =============================================================================

print("\n")
print("=" * 90)
print("PAIRED DELONG TESTS")
print("=" * 90)

results = []

comparisons = [

    (
        "SegFormer vs YOLO_referee",
        segformer_failure_score,

        yolo_failure_score,

        segformer_auc,

        yolo_auc
    ),

]

for (

    comparison,
    score1,
    score2,
    auc1,
    auc2,

) in comparisons:

    log10_p = delong_roc_test(

        y_true,
        score1,
        score2

    )

    log10_p = float(
        np.asarray(log10_p).squeeze()
    )

    p_value = 10 ** log10_p

    auc_difference = auc1 - auc2

    significant = (
        "Yes"
        if p_value < 0.05
        else "No"
    )

    print("\n" + "-" * 70)

    print(comparison)

    print(f"AUC 1          : {auc1:.6f}")
    print(f"AUC 2          : {auc2:.6f}")
    print(f"Difference     : {auc_difference:.6f}")
    print(f"log10(p)       : {log10_p:.6f}")
    print(f"p-value        : {p_value:.12f}")
    print(f"Significant    : {significant}")

    results.append({

        "Comparison": comparison,

        "Method_1_AUC": round(auc1, 6),

        "Method_2_AUC": round(auc2, 6),

        "AUC_Difference": round(
            auc_difference,
            6
        ),

        "log10_p": round(
            log10_p,
            6
        ),

        "p_value": p_value,

        "Significant": significant,

    })

# =============================================================================
# CREATE RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(results)

auc_summary = pd.DataFrame({

    "Method": [
       "SegFormer",
        "YOLO Referee"
    ],

    "ROC_AUC": [
        round(segformer_auc,6),
        round(yolo_auc,6)
    ]

})

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print("\nROC-AUC Summary\n")
print(auc_summary.to_string(index=False))

print("\nPaired DeLong Results\n")
print(results_df.to_string(index=False))

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n")
print("=" * 90)
print("SAVING RESULTS")
print("=" * 90)

# -------------------------------------------------------------------------
# Output Paths
# -------------------------------------------------------------------------

auc_summary_path = os.path.join(
    OUTPUT_DIR,
    "paired_auc_summary.csv"
)

delong_path = os.path.join(
    OUTPUT_DIR,
    "paired_delong_results.csv"
)

merged_path = os.path.join(
    OUTPUT_DIR,
    "merged_referee_scores.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "paired_delong_summary.txt"
)

# -------------------------------------------------------------------------
# Save CSV Files
# -------------------------------------------------------------------------

auc_summary.to_csv(
    auc_summary_path,
    index=False
)

results_df.to_csv(
    delong_path,
    index=False
)

merged_df.to_csv(
    merged_path,
    index=False
)

# -------------------------------------------------------------------------
# Save Reviewer Summary
# -------------------------------------------------------------------------

with open(summary_path, "w") as f:

    f.write("=" * 80 + "\n")
    f.write("PAIRED DELONG COMPARISON OF REFEREE METHODS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total Images : {len(merged_df)}\n")
    f.write(
        f"Failure Threshold : Dice < {FAILURE_THRESHOLD:.2f}\n\n"
    )

    f.write("ROC-AUC SUMMARY\n")
    f.write("-" * 80 + "\n")

    f.write(
        auc_summary.to_string(index=False)
    )

    f.write("\n\n")

    f.write("PAIRED DELONG RESULTS\n")
    f.write("-" * 80 + "\n")

    f.write(
        results_df.to_string(index=False)
    )

print("\nFiles Saved")
print("-" * 40)

print(auc_summary_path)
print(delong_path)
print(merged_path)
print(summary_path)

# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================

print("\n")
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print("\nROC-AUC")
print("-" * 40)

print(f"SegFormer     : {segformer_auc:.6f}")
print(f"YOLO Referee  : {yolo_auc:.6f}")

print("\nPaired DeLong Results")
print("-" * 40)

print(results_df.to_string(index=False))

print("\n")
print("=" * 90)
print("PAIRED DELONG ANALYSIS COMPLETE")
print("=" * 90)