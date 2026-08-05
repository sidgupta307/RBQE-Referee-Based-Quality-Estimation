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

YOLO_MASK_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\02_external_predictions"
)

REFEREE_MASK_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\03_external_referee_predictions"
)

PHASE4_METRICS = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_03_UNetPP_1223"
    r"\merged_1223.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\04_yolo_seed_agreement"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# HELPERS
# ============================================================

def dice_score(a, b):

    a = a.astype(bool)
    b = b.astype(bool)

    inter = np.logical_and(a, b).sum()

    return (
        2.0 * inter
    ) / (
        a.sum() + b.sum() + 1e-8
    )


def iou_score(a, b):

    a = a.astype(bool)
    b = b.astype(bool)

    inter = np.logical_and(a, b).sum()

    union = np.logical_or(a, b).sum()

    return inter / (union + 1e-8)


def centroid(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return np.array([
        xs.mean(),
        ys.mean()
    ])


def normalized_centroid_distance(
    mask1,
    mask2
):

    c1 = centroid(mask1)
    c2 = centroid(mask2)

    if c1 is None or c2 is None:
        return 1.0

    H, W = mask1.shape

    diag = np.sqrt(
        H * H + W * W
    )

    dist = np.linalg.norm(
        c1 - c2
    )

    return dist / diag


def area_ratio(
    mask1,
    mask2
):

    a1 = np.sum(mask1)
    a2 = np.sum(mask2)

    if max(a1, a2) == 0:
        return 1.0

    return (
        min(a1, a2)
        /
        max(a1, a2)
    )


def boundary_mask(mask):

    mask = mask.astype(bool)

    eroded = binary_erosion(
        mask
    )

    boundary = (
        mask ^ eroded
    )

    return boundary.astype(
        np.uint8
    )


# ============================================================
# LOAD PHASE 4 METRICS
# ============================================================

metrics_df = pd.read_csv(
    PHASE4_METRICS
)

print(
    f"\nLoaded Phase 4 metrics: "
    f"{len(metrics_df)} images"
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

rows = []

for _, row in tqdm(
    metrics_df.iterrows(),
    total=len(metrics_df)
):

    stem = str(row["image"])

    yolo_path = os.path.join(
        YOLO_MASK_DIR,
        row["dataset"],
        "masks",
        stem + ".png"
    )

    referee_path = os.path.join(
        REFEREE_MASK_DIR,
        row["dataset"],
        "masks",
        stem + ".png"
    )

    if not os.path.exists(yolo_path):
        continue

    if not os.path.exists(referee_path):
        continue

    yolo = cv2.imread(
        yolo_path,
        0
    )

    referee = cv2.imread(
        referee_path,
        0
    )

    yolo = (
        yolo > 0
    ).astype(np.uint8)

    referee = (
        referee > 0
    ).astype(np.uint8)

    # ====================================
    # FEATURES
    # ====================================

    agreement_dice = dice_score(
        yolo,
        referee
    )

    agreement_iou = iou_score(
        yolo,
        referee
    )

    area_match = area_ratio(
        yolo,
        referee
    )

    centroid_dist = (
        normalized_centroid_distance(
            yolo,
            referee
        )
    )

    boundary_yolo = boundary_mask(
        yolo
    )

    boundary_referee = boundary_mask(
    referee
    )

    boundary_agreement = dice_score(
        boundary_yolo,
        boundary_referee
    )

    new_row = row.copy()

    new_row["agreement_dice"] = agreement_dice
    new_row["agreement_iou"] = agreement_iou
    new_row["area_ratio"] = area_match
    new_row["centroid_distance"] = centroid_dist
    new_row["boundary_agreement"] = boundary_agreement

    rows.append(new_row)

# ============================================================
# SAVE FEATURES
# ============================================================

features_df = pd.DataFrame(
    rows
)

# ------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------

print("\nRows written:", len(features_df))

assert len(features_df) == len(metrics_df), \
    "ERROR: Some benchmark images were skipped during processing."

feature_csv = os.path.join(
    OUTPUT_DIR,
    "merged_1223_yolo_seed.csv"
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
# FEATURE RANKING
# ============================================================

feature_cols = [

    "agreement_dice",
    "agreement_iou",
    "area_ratio",
    "centroid_distance",
    "boundary_agreement"

]

ranking = []

for feat in feature_cols:

    rho, p = spearmanr(

        features_df[feat],
        features_df["dice"]

    )

    ranking.append({

        "feature": feat,
        "spearman_rho": rho,
        "p_value": p

    })

ranking_df = (
    pd.DataFrame(ranking)
    .sort_values(
        "spearman_rho",
        ascending=False
    )
)

ranking_csv = os.path.join(
    OUTPUT_DIR,
    "feature_ranking.csv"
)

ranking_df.to_csv(
    ranking_csv,
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("AGREEMENT FEATURE RANKING")
print("=" * 70)

print(
    ranking_df
)

print("\n")
print("=" * 70)
print("YOLO SEED AGREEMENT EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Images: {len(features_df)}"
)

print(
    f"Features: {len(feature_cols)}"
)