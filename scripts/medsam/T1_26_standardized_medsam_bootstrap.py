"""
==============================================================
T1_26_standardized_medsam_bootstrap.py
==============================================================

Bootstrap 95% confidence intervals for standardized
Independent MedSAM evaluation (1223-image benchmark)

Inputs
------
clean_agreement_features.csv (1223 benchmark)
agreement_features_independent.csv
combined_results.csv

Output
------
bootstrap_auc_results.csv

==============================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

# ============================================================
# INPUTS
# ============================================================

MASTER_CSV = (
    r"C:\seg_uncertain\chat_gpt\20B_segformer_clean_agreement"
    r"\clean_agreement_features.csv"
)

MEDSAM_CSV = (
    r"C:\seg_uncertain\chat_gpt\29_medsam_independent_comparison"
    r"\agreement_features_independent.csv"
)

YOLO_RESULTS = (
    r"C:\seg_uncertain\chat_gpt\01_external_validation"
    r"\combined_results.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_04_MedSAM_1223"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOTSTRAPS = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ============================================================
# LOAD
# ============================================================

master = pd.read_csv(MASTER_CSV)
medsam = pd.read_csv(MEDSAM_CSV)
results = pd.read_csv(YOLO_RESULTS)

results = results[
    results["model"] == "YOLOv8-Seg"
].copy()

# ============================================================
# NORMALIZE IMAGE NAMES
# ============================================================

def normalize(df):

    df = df.copy()

    df["image"] = (
        df["image"]
        .str.lower()
        .str.strip()
        .str.replace(".png", "", regex=False)
        .str.replace(".jpg", "", regex=False)
    )

    return df


master = normalize(master)
medsam = normalize(medsam)
results = normalize(results)

master = master[
    [
        "dataset",
        "image"
    ]
]

merged = master.merge(
    medsam,
    on=[
        "dataset",
        "image"
    ],
    how="inner"
)

merged = merged.merge(
    results[
        [
            "dataset",
            "image",
            "dice"
        ]
    ],
    on=[
        "dataset",
        "image"
    ],
    how="inner"
)

assert len(merged) == 1223

# ============================================================
# FAILURE LABEL
# ============================================================

merged["failure"] = (
    merged["dice"] < 0.50
).astype(int)

# ============================================================
# FEATURES
# ============================================================

FEATURES = {
    "Agreement Dice": "agreement_dice",
    "Agreement IoU": "agreement_iou",
    "Area Ratio": "area_ratio",
    "Boundary Agreement": "boundary_agreement",
    "Centroid Distance": "centroid_distance",
}

results_list = []

# ============================================================
# BOOTSTRAP
# ============================================================

for signal_name, feature in FEATURES.items():

    df = merged[
        [feature, "failure"]
    ].dropna()

    y = df["failure"].values

    x = df[feature].values

    if feature == "centroid_distance":
        scores = x
    else:
        scores = -x

    aucs = []

    n = len(df)

    for _ in range(N_BOOTSTRAPS):

        idx = np.random.choice(
            n,
            n,
            replace=True
        )

        y_boot = y[idx]
        s_boot = scores[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        auc = roc_auc_score(
            y_boot,
            s_boot
        )

        aucs.append(auc)

    aucs = np.asarray(aucs)

    results_list.append({

        "Signal": signal_name,

        "Bootstrap_Mean":
            aucs.mean(),

        "Bootstrap_STD":
            aucs.std(ddof=1),

        "CI_Low":
            np.percentile(
                aucs,
                2.5
            ),

        "CI_High":
            np.percentile(
                aucs,
                97.5
            )

    })

# ============================================================
# SAVE
# ============================================================

results_df = (
    pd.DataFrame(results_list)
    .sort_values(
        "Bootstrap_Mean",
        ascending=False
    )
)

save_csv = os.path.join(
    OUTPUT_DIR,
    "bootstrap_auc_results.csv"
)

results_df.to_csv(
    save_csv,
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print()
print(results_df.to_string(index=False))

print()
print("Saved to:")
print(save_csv)

print()
print("Done.")