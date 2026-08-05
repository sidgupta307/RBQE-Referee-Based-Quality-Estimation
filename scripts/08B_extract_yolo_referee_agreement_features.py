import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.ndimage import binary_erosion

# ============================================================
# PATHS
# ============================================================

STACK_DIR = (
    r"C:\seg_uncertain\segformer_uncertainty\mask_stacks"
)

CONSENSUS_DIR = (
    r"C:\seg_uncertain\segformer_uncertainty\consensus_masks"
)

VARIANCE_DIR = (
    r"C:\seg_uncertain\segformer_uncertainty\variance_maps"
)

METRIC_CSV = (
    r"C:\seg_uncertain\segformer_test_evaluation"
    r"\per_image_metrics.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\segformer_uncertainty_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# HELPERS
# ============================================================

def boundary_mask(mask):

    mask = mask.astype(np.uint8)

    kernel = np.ones(
        (5, 5),
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
# LOAD TRUE METRICS
# ============================================================

metrics_df = pd.read_csv(
    METRIC_CSV
)

dice_lookup = {}

for _, row in metrics_df.iterrows():

    dice_lookup[
        os.path.splitext(
            row["image"]
        )[0]
    ] = row["dice"]

print(
    "Loaded Dice:",
    len(dice_lookup)
)

# ============================================================
# EXTRACT FEATURES
# ============================================================

rows = []

stack_files = [

    f for f in os.listdir(
        STACK_DIR
    )

    if f.endswith(".npy")
]

for file in tqdm(stack_files):

    stem = os.path.splitext(
        file
    )[0]

    stack_path = os.path.join(
        STACK_DIR,
        file
    )

    stack = np.load(
        stack_path
    )

    consensus = np.mean(
        stack,
        axis=0
    )

    variance = np.var(
        stack,
        axis=0
    )
    variance = variance / 0.25

    consensus_mask = (
        consensus > 0.5
    ).astype(np.uint8)

    boundary = boundary_mask(
        consensus_mask
    )

    unc_mean = float(
        variance.mean()
    )

    unc_std = float(
        variance.std()
    )

    unc_max = float(
        variance.max()
    )

    foreground_unc = float(
        variance[
            consensus_mask == 1
        ].mean()
    ) if consensus_mask.sum() > 0 else 0

    background_unc = float(
        variance[
            consensus_mask == 0
        ].mean()
    )

    boundary_unc = float(
        variance[
            boundary == 1
        ].mean()
    ) if boundary.sum() > 0 else 0

    disagreement = np.logical_and(

        consensus > 0,

        consensus < 1

    )

    pixel_disagreement_ratio = float(

        disagreement.sum()

        /

        disagreement.size

    )

    consensus_score = float(

        np.mean(

            np.abs(
                consensus - 0.5
            )

        )

    )

    consensus_area = int(
        consensus_mask.sum()
    )

    num_zero_predictions = int(

        np.sum(

            stack.reshape(
                stack.shape[0],
                -1
            ).sum(axis=1) == 0

        )

    )

    rows.append({

        "image":
            stem,

        "gt_dice":
            dice_lookup.get(
                stem,
                np.nan
            ),

        "unc_mean":
            unc_mean,

        "unc_std":
            unc_std,

        "unc_max":
            unc_max,

        "foreground_unc":
            foreground_unc,

        "background_unc":
            background_unc,

        "boundary_unc":
            boundary_unc,

        "pixel_disagreement_ratio":
            pixel_disagreement_ratio,

        "consensus_score":
            consensus_score,

        "consensus_area":
            consensus_area,

        "num_zero_predictions":
            num_zero_predictions

    })

# ============================================================
# SAVE FEATURES
# ============================================================

features_df = pd.DataFrame(
    rows
)

feature_csv = os.path.join(
    OUTPUT_DIR,
    "segformer_uncertainty_features_aligned.csv"
)

features_df.to_csv(
    feature_csv,
    index=False
)

print(
    "\nSaved:"
)
print(
    feature_csv
)

# ============================================================
# CORRELATIONS
# ============================================================

feature_cols = [

    "unc_mean",
    "unc_std",
    "unc_max",
    "foreground_unc",
    "background_unc",
    "boundary_unc",
    "pixel_disagreement_ratio",
    "consensus_score",
    "consensus_area",
    "num_zero_predictions"

]

ranking = []

for feat in feature_cols:

    rho, p = spearmanr(

        features_df[feat],
        features_df["gt_dice"]

    )

    ranking.append({

        "feature":
            feat,

        "spearman_rho":
            rho,

        "p_value":
            p

    })

ranking_df = pd.DataFrame(
    ranking
)

ranking_df = ranking_df.sort_values(
    "spearman_rho",
    ascending=False
)

ranking_csv = os.path.join(
    OUTPUT_DIR,
    "feature_correlations.csv"
)

ranking_df.to_csv(
    ranking_csv,
    index=False
)

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("SEGFORMER UNCERTAINTY FEATURE RANKING")
print("=" * 70)

print(ranking_df)

print("\n")
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Images:",
    len(features_df)
)