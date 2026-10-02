import os
import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\11_non_empty_robustness"
    r"\non_empty_evaluation.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\11_non_empty_robustness"
    r"\bootstrap"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# BOOTSTRAP SETTINGS
# ============================================================

N_BOOTSTRAPS = 2000

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

rng = np.random.RandomState(
    RANDOM_SEED
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print("=" * 70)
print("BOOTSTRAP ROC-AUC ANALYSIS")
print("=" * 70)

print("Experiment      : Non-empty mask robustness")
print("Evaluation set  : Images where BOTH models produced non-empty masks")
print()

print(
    f"Images Loaded  : {len(df)}"
)

print(
    f"CSV Source     : {INPUT_CSV}"
)

print()

EXPECTED_IMAGES = 975

if len(df) != EXPECTED_IMAGES:
    raise ValueError(
        f"Expected {EXPECTED_IMAGES} images, "
        f"but found {len(df)}."
    )

print("Dataset verification passed.")
print()

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
    f"{len(df)-df['failure'].sum()}"
)

# ============================================================
# FEATURES
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
# BOOTSTRAP SINGLE FEATURE
# ============================================================

def bootstrap_auc(

    dataframe,
    feature_name,
    feature_column

):

    temp = dataframe[
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
    # Higher score must indicate HIGHER failure risk
    # --------------------------------------------------------

    if feature_column != "centroid_distance":

        scores = -scores

    # --------------------------------------------------------
    # ORIGINAL ROC-AUC
    # --------------------------------------------------------

    original_auc = roc_auc_score(

        y_true,
        scores

    )

    # --------------------------------------------------------
    # BOOTSTRAP
    # --------------------------------------------------------

    bootstrap_scores = []

    n = len(
        y_true
    )

    while len(bootstrap_scores) < N_BOOTSTRAPS:

        indices = rng.randint(

            0,
            n,
            n

        )

        # Skip invalid samples
        if len(
            np.unique(
                y_true[indices]
            )
        ) < 2:

            continue

        auc = roc_auc_score(

            y_true[indices],
            scores[indices]

        )

        bootstrap_scores.append(
            auc
        )

    bootstrap_scores = np.array(
        bootstrap_scores
    )

    assert len(bootstrap_scores) == N_BOOTSTRAPS

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------

    ci_lower = np.percentile(

        bootstrap_scores,
        2.5

    )

    ci_upper = np.percentile(

        bootstrap_scores,
        97.5

    )

    mean_auc = np.mean(
        bootstrap_scores
    )

    std_auc = np.std(

        bootstrap_scores,
        ddof=1

    )

    return {

        "Signal":
            feature_name,

        "Feature":
            feature_column,

        "ROC_AUC":
            original_auc,

        "Bootstrap_Mean":
            mean_auc,

        "CI_Lower":
            ci_lower,

        "CI_Upper":
            ci_upper,

        "Bootstrap_STD":
            std_auc

    }

# ============================================================
# RUN BOOTSTRAP FOR ALL FEATURES
# ============================================================

print()
print("=" * 70)
print("BOOTSTRAP RESULTS")
print("=" * 70)

for feature_name, feature_column in FEATURES.items():

    result = bootstrap_auc(

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

output_csv = os.path.join(

    OUTPUT_DIR,

    "bootstrap_auc_results.csv"

)

results_df.to_csv(

    output_csv,

    index=False

)

# ============================================================
# SUMMARY
# ============================================================

best = results_df.iloc[0]

print()
print("=" * 70)
print("BEST BOOTSTRAP RESULT")
print("=" * 70)

print(
    f"Signal          : {best['Signal']}"
)

print(
    f"ROC-AUC         : {best['ROC_AUC']:.6f}"
)

print(
    f"Bootstrap Mean  : {best['Bootstrap_Mean']:.6f}"
)

print(
    f"95% CI          : "
    f"[{best['CI_Lower']:.6f}, "
    f"{best['CI_Upper']:.6f}]"
)

print(
    f"Bootstrap STD   : {best['Bootstrap_STD']:.6f}"
)

print()
print("=" * 70)
print("NON-EMPTY MASK BOOTSTRAP ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Features Evaluated : {len(results_df)}"
)

print(
    f"Bootstrap Samples  : {N_BOOTSTRAPS}"
)

print()

print("Results saved to:")

print(output_csv)