import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================
# INPUT
# ============================================================

INPUT_CSV = r"C:\seg_uncertain\journal_extension\T1_03_UNetPP_1223\merged_1223.csv"

OUTPUT_DIR = r"C:\seg_uncertain\journal_extension\T1_03_UNetPP_1223"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

FEATURES = {
    "Agreement Dice": "agreement_dice",
    "Agreement IoU": "agreement_iou",
    "Area Ratio": "area_ratio",
    "Boundary Agreement": "boundary_agreement",
    "Centroid Distance": "centroid_distance"
}

N_BOOT = 1000

np.random.seed(42)

results = []

# ============================================================
# BOOTSTRAP
# ============================================================

for signal, feature in FEATURES.items():

    temp = df[[feature, "failure"]].dropna()

    y = temp["failure"].values

    score = temp[feature].values.copy()

    # Higher score = higher failure probability
    if feature != "centroid_distance":
        score = -score

    aucs = []

    n = len(temp)

    for _ in range(N_BOOT):

        idx = np.random.choice(
            n,
            n,
            replace=True
        )

        y_boot = y[idx]

        s_boot = score[idx]

        # ROC undefined if only one class present
        if len(np.unique(y_boot)) < 2:
            continue

        aucs.append(
            roc_auc_score(
                y_boot,
                s_boot
            )
        )

    aucs = np.array(aucs)

    results.append({

        "Signal": signal,

        "Bootstrap_Mean": aucs.mean(),

        "Bootstrap_STD": aucs.std(),

        "CI_Low": np.percentile(aucs, 2.5),

        "CI_High": np.percentile(aucs, 97.5)

    })

results = pd.DataFrame(results)

results = results.sort_values(
    "Bootstrap_Mean",
    ascending=False
)

print()
print(results)

results.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "bootstrap_auc_results.csv"
    ),

    index=False

)

print("\nDone.")