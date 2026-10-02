import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# PATHS
# ============================================================

AGREEMENT_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\24_unetpp_agreement\agreement_features.csv"
)

RESULTS_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\01_external_validation\combined_results.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\24A_unetpp_agreement_fixed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD
# ============================================================

agreement_df = pd.read_csv(
    AGREEMENT_CSV
)

results_df = pd.read_csv(
    RESULTS_CSV
)

results_df = results_df[
    results_df["model"] == "YOLOv8-Seg"
].copy()

# ============================================================
# IMAGE KEY
# ============================================================

agreement_df["image_key"] = (
    agreement_df["image"]
    .str.replace(".jpg","",regex=False)
    .str.replace(".png","",regex=False)
)

results_df["image_key"] = (
    results_df["image"]
    .str.replace(".jpg","",regex=False)
    .str.replace(".png","",regex=False)
)

# ============================================================
# MERGE
# ============================================================

merged = pd.merge(
    agreement_df,
    results_df[
        [
            "dataset",
            "image_key",
            "dice"
        ]
    ],
    on=[
        "dataset",
        "image_key"
    ]
)

print(
    "\nMerged Images:",
    len(merged)
)

# ============================================================
# CORRELATIONS
# ============================================================

features = [
    "agreement_dice",
    "agreement_iou",
    "area_ratio",
    "centroid_distance",
    "boundary_agreement"
]

rows = []

for f in features:
    

    rho, p = spearmanr(
    merged[f],
    merged["dice"],
    nan_policy="omit"
    )

    rows.append([
        f,
        rho,
        p
    ])

corr_df = pd.DataFrame(
    rows,
    columns=[
        "feature",
        "spearman_rho",
        "p_value"
    ]
)

corr_df = corr_df.sort_values(
    "spearman_rho",
    ascending=False
)

corr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_correlations.csv"
    ),
    index=False
)

print("\n")
print("="*70)
print("UNET++ CORRELATION (FIXED)")
print("="*70)

print(corr_df)

print("\nSaved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "feature_correlations.csv"
    )
)