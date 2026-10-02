import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# PATHS
# ============================================================

INPUT_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\05_agreement_features"
    r"\agreement_features.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\13_clean_agreement"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print("\nOriginal Images:", len(df))

# ============================================================
# REMOVE BOTH-EMPTY
# ============================================================
#
# Agreement Dice == 1
# AND centroid_distance == 1
#
# From previous diagnostics:
# these correspond to
# YOLO empty + MedSAM empty
#
# ============================================================

both_empty = (

    (df["agreement_dice"] >= 0.9999)
    &
    (df["centroid_distance"] >= 0.9999)

)

print(
    "Both Empty Removed:",
    both_empty.sum()
)

clean_df = df[
    ~both_empty
].copy()

print(
    "Remaining Images:",
    len(clean_df)
)

# ============================================================
# SAVE
# ============================================================

clean_csv = os.path.join(
    OUTPUT_DIR,
    "clean_agreement_features.csv"
)

clean_df.to_csv(
    clean_csv,
    index=False
)

# ============================================================
# RECOMPUTE CORRELATIONS
# ============================================================

features = [

    "agreement_dice",
    "agreement_iou",
    "area_ratio",
    "boundary_agreement",
    "centroid_distance"

]

rows = []

for feat in features:

    rho, p = spearmanr(
        clean_df[feat],
        clean_df["gt_dice"]
    )

    rows.append({

        "feature":
            feat,

        "spearman_rho":
            rho,

        "p_value":
            p

    })

corr_df = pd.DataFrame(
    rows
)

corr_df = corr_df.sort_values(
    "spearman_rho",
    ascending=False
)

corr_csv = os.path.join(
    OUTPUT_DIR,
    "clean_correlations.csv"
)

corr_df.to_csv(
    corr_csv,
    index=False
)

# ============================================================
# PRINT
# ============================================================

print("\n")
print("="*70)
print("CLEAN AGREEMENT CORRELATIONS")
print("="*70)

print(corr_df)

print("\nSaved:")
print(clean_csv)
print(corr_csv)