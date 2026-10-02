import os
import numpy as np
import pandas as pd

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
    r"\10_risk_coverage"
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
print("RISK-COVERAGE ANALYSIS")
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

if len(missing):

    raise ValueError(
        f"Missing columns: {missing}"
    )

print("All required columns found.")

print()

print(
    f"Mean Dice : {df['dice'].mean():.6f}"
)

print(
    f"Median Dice : {df['dice'].median():.6f}"
)

print(
    f"Minimum Dice : {df['dice'].min():.6f}"
)

print(
    f"Maximum Dice : {df['dice'].max():.6f}"
)

# ============================================================
# AGREEMENT FEATURES
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
# COVERAGE LEVELS
# ============================================================

COVERAGE_LEVELS = np.arange(

    1.00,
    0.09,
    -0.05

)

results = []

curve_data = {}

print()

print("=" * 70)
print("DATA VALIDATION COMPLETE")
print("=" * 70)

# ============================================================
# RISK-COVERAGE COMPUTATION
# ============================================================

print()
print("=" * 70)
print("RUNNING RISK-COVERAGE ANALYSIS")
print("=" * 70)

for feature_name, feature_column in FEATURES.items():

    print()
    print(feature_name)

    temp = df.copy()

    # --------------------------------------------------------
    # Convert every signal so that
    # larger value = better confidence
    # --------------------------------------------------------

    if feature_column == "centroid_distance":

        confidence = -temp[feature_column].values

    else:

        confidence = temp[feature_column].values

    temp["confidence"] = confidence

    # --------------------------------------------------------
    # Highest-confidence images first
    # --------------------------------------------------------

    temp = temp.sort_values(

        "confidence",

        ascending=False

    ).reset_index(

        drop=True

    )

    coverage_list = []
    mean_dice_list = []

    for coverage in COVERAGE_LEVELS:

        keep = max(

            1,

            int(
                np.ceil(
                    coverage * len(temp)
                )
            )

        )

        accepted = temp.iloc[:keep]

        mean_dice = accepted[
            "dice"
        ].mean()

        coverage_list.append(
            coverage
        )

        mean_dice_list.append(
            mean_dice
        )

        results.append({

            "Signal":
                feature_name,

            "Feature":
                feature_column,

            "Coverage":
                coverage,

            "Accepted_Images":
                keep,

            "Rejected_Images":
                len(temp) - keep,

            "Mean_Dice":
                mean_dice

        })

    curve_data[
        feature_name
    ] = {

        "coverage":
            coverage_list,

        "mean_dice":
            mean_dice_list

    }

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(

    by=[
        "Signal",
        "Coverage"
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

print(results_df.head(25))

# ============================================================
# SAVE RESULTS
# ============================================================

results_csv = os.path.join(

    OUTPUT_DIR,

    "risk_coverage_results.csv"

)

results_df.to_csv(

    results_csv,

    index=False

)

# ============================================================
# COVERAGE SUMMARY
# ============================================================

summary_df = (

    results_df
    .groupby("Signal")["Mean_Dice"]
    .agg(
        Mean="mean",
        Maximum="max",
        Minimum="min"
    )
    .reset_index()

)

summary_csv = os.path.join(

    OUTPUT_DIR,

    "coverage_summary.csv"

)

summary_df.to_csv(

    summary_csv,

    index=False

)

# ============================================================
# RISK-COVERAGE CURVES
# ============================================================

plt.figure(figsize=(8,6))

for feature_name in FEATURES.keys():

    plt.plot(

        curve_data[feature_name]["coverage"],

        curve_data[feature_name]["mean_dice"],

        marker="o",

        linewidth=2,

        label=feature_name

    )

plt.xlabel("Coverage")

plt.ylabel("Mean Dice")

plt.title("Risk-Coverage Curves")

plt.grid(True)

plt.legend()

plt.xlim(1.0, 0.10)

plt.tight_layout()

figure_path = os.path.join(

    OUTPUT_DIR,

    "risk_coverage_curves.png"

)

plt.savefig(

    figure_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ============================================================
# BEST FEATURE AT EACH COVERAGE
# ============================================================

best_df = (

    results_df
    .sort_values(
        ["Coverage","Mean_Dice"],
        ascending=[False,False]
    )
    .groupby(
        "Coverage",
        as_index=False
    )
    .first()

)

print()

print("="*70)
print("BEST FEATURE PER COVERAGE")
print("="*70)

print(

    best_df[
        [
            "Coverage",
            "Signal",
            "Mean_Dice",
            "Accepted_Images",
            "Rejected_Images"
        ]
    ]

)

# ============================================================
# OVERALL BEST FEATURE
# ============================================================

overall = (

    results_df
    .groupby("Signal")["Mean_Dice"]
    .mean()
    .sort_values(
        ascending=False
    )

)

print()

print("="*70)
print("OVERALL MEAN DICE")
print("="*70)

print(overall)

best_signal = overall.index[0]

print()

print("="*70)
print("BEST OVERALL SIGNAL")
print("="*70)

print(f"Signal : {best_signal}")
print(f"Mean Dice : {overall.iloc[0]:.6f}")

# ============================================================
# SUMMARY
# ============================================================

print()

print("="*70)
print("RISK-COVERAGE ANALYSIS COMPLETE")
print("="*70)

print(
    f"Images Evaluated : {len(df)}"
)

print(
    f"Coverage Levels : {len(COVERAGE_LEVELS)}"
)

print(
    f"Signals Tested : {len(FEATURES)}"
)

print(
    f"Total Evaluations : {len(results_df)}"
)

print()

print("Results saved to:")

print(results_csv)

print(summary_csv)

print(figure_path)