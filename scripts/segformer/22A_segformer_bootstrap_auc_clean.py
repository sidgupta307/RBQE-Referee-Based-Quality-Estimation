import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

# ============================================================
# PATHS
# ============================================================

INPUT_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\20B_segformer_clean_agreement"
    r"\clean_agreement_features.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\22A_segformer_bootstrap_auc"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# CONFIG
# ============================================================

N_BOOTSTRAPS = 1000

FAILURE_THRESHOLD = 0.50

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    "\nMerged Images:",
    len(df)
)

df["failure"] = (
    df["gt_dice"] < FAILURE_THRESHOLD
).astype(int)

print(
    "Failures:",
    df["failure"].sum()
)

print(
    "Non Failures:",
    len(df) - df["failure"].sum()
)

# ============================================================
# SIGNALS
#
# Higher value must mean
# MORE LIKELY FAILURE
# ============================================================

signals = {

    "Agreement Dice":
        1.0 - df["agreement_dice"].values,

    "Agreement IoU":
        1.0 - df["agreement_iou"].values,

    "Area Ratio":
        1.0 - df["area_ratio"].values,

    "Boundary Agreement":
        1.0 - df["boundary_agreement"].values,

    "Centroid Distance":
        df["centroid_distance"].values

}

y_true = df["failure"].values

# ============================================================
# BOOTSTRAP
# ============================================================

results = []

for signal_name, scores in signals.items():

    print(
        f"\nBootstrapping: {signal_name}"
    )

    base_auc = roc_auc_score(
        y_true,
        scores
    )

    bootstrap_aucs = []

    for _ in range(
        N_BOOTSTRAPS
    ):

        idx = np.random.choice(
            len(y_true),
            size=len(y_true),
            replace=True
        )

        y_boot = y_true[idx]
        s_boot = scores[idx]

        # Need both classes
        if len(np.unique(y_boot)) < 2:
            continue

        auc = roc_auc_score(
            y_boot,
            s_boot
        )

        bootstrap_aucs.append(
            auc
        )

    bootstrap_aucs = np.array(
        bootstrap_aucs
    )

    results.append({

        "Signal":
            signal_name,

        "AUC":
            base_auc,

        "Bootstrap_Mean":
            bootstrap_aucs.mean(),

        "Bootstrap_STD":
            bootstrap_aucs.std(),

        "CI_Low":
            np.percentile(
                bootstrap_aucs,
                2.5
            ),

        "CI_High":
            np.percentile(
                bootstrap_aucs,
                97.5
            )

    })

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "AUC",
    ascending=False
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "bootstrap_auc_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 80)
print("SEGFORMER BOOTSTRAP AUC RESULTS")
print("=" * 80)

print(results_df)

print("\nSaved:")
print(csv_path)