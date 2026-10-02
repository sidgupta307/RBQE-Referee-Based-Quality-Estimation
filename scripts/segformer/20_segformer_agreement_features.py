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

YOLO_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\02_external_predictions"
)

SEGFORMER_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\19_segformer_external_predictions"
)

METRICS_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\01_external_validation"
    r"\combined_results.csv"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\20_segformer_agreement"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# METRICS
# ============================================================

def dice_score(mask1, mask2):

    inter = np.logical_and(
        mask1,
        mask2
    ).sum()

    return (
        2.0 * inter
    ) / (
        mask1.sum()
        + mask2.sum()
        + 1e-8
    )


def iou_score(mask1, mask2):

    inter = np.logical_and(
        mask1,
        mask2
    ).sum()

    union = np.logical_or(
        mask1,
        mask2
    ).sum()

    return inter / (
        union + 1e-8
    )


def area_ratio(mask1, mask2):

    a1 = mask1.sum()
    a2 = mask2.sum()

    if max(a1, a2) == 0:
        return 1.0

    return min(a1, a2) / max(a1, a2)


def centroid(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return np.array([
        xs.mean(),
        ys.mean()
    ])


def centroid_distance(mask1, mask2):

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


def boundary_mask(mask):

    eroded = binary_erosion(mask)

    return (
        mask.astype(np.uint8)
        ^
        eroded.astype(np.uint8)
    )


# ============================================================
# LOAD YOLO PERFORMANCE ONLY
# ============================================================

metrics_df = pd.read_csv(
    METRICS_CSV
)

metrics_df = metrics_df[
    metrics_df["model"] == "YOLOv8-Seg"
].reset_index(drop=True)

print(
    "\nYOLO Images:",
    len(metrics_df)
)

# ============================================================
# EXTRACT FEATURES
# ============================================================

rows = []

for _, row in tqdm(
    metrics_df.iterrows(),
    total=len(metrics_df)
):

    dataset = row["dataset"]
    image = row["image"]

    stem = os.path.splitext(
        image
    )[0]

    yolo_path = os.path.join(
        YOLO_ROOT,
        dataset,
        "masks",
        stem + ".png"
    )

    segformer_path = os.path.join(
        SEGFORMER_ROOT,
        dataset,
        "masks",
        stem + ".png"
    )

    if not os.path.exists(yolo_path):
        continue

    if not os.path.exists(segformer_path):
        continue

    yolo = cv2.imread(
        yolo_path,
        0
    )

    segformer = cv2.imread(
        segformer_path,
        0
    )

    if yolo is None:
        continue

    if segformer is None:
        continue

    yolo = (
        yolo > 0
    ).astype(np.uint8)

    segformer = (
        segformer > 0
    ).astype(np.uint8)

    agreement_dice = dice_score(
        yolo,
        segformer
    )

    agreement_iou = iou_score(
        yolo,
        segformer
    )

    ar = area_ratio(
        yolo,
        segformer
    )

    cd = centroid_distance(
        yolo,
        segformer
    )

    boundary_yolo = boundary_mask(
        yolo
    )

    boundary_segformer = boundary_mask(
        segformer
    )

    boundary_agreement = dice_score(
        boundary_yolo,
        boundary_segformer
    )

    rows.append({

        "dataset":
            dataset,

        "image":
            image,

        "gt_dice":
            row["dice"],

        "agreement_dice":
            agreement_dice,

        "agreement_iou":
            agreement_iou,

        "area_ratio":
            ar,

        "centroid_distance":
            cd,

        "boundary_agreement":
            boundary_agreement

    })

# ============================================================
# SAVE FEATURES
# ============================================================

features_df = pd.DataFrame(
    rows
)

feature_csv = os.path.join(
    OUTPUT_DIR,
    "agreement_features.csv"
)

features_df.to_csv(
    feature_csv,
    index=False
)

print("\nSaved:")
print(feature_csv)

# ============================================================
# CORRELATIONS
# ============================================================

ranking = []

feature_cols = [

    "agreement_dice",
    "agreement_iou",
    "area_ratio",
    "centroid_distance",
    "boundary_agreement"

]

for feature in feature_cols:

    rho, p = spearmanr(

        features_df[feature],
        features_df["gt_dice"]

    )

    ranking.append({

        "feature":
            feature,

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

corr_csv = os.path.join(
    OUTPUT_DIR,
    "feature_correlations.csv"
)

ranking_df.to_csv(
    corr_csv,
    index=False
)

# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 70)
print("YOLO + SEGFORMER AGREEMENT FEATURE RANKING")
print("=" * 70)

print(
    ranking_df
)

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    "Images:",
    len(features_df)
)

print(
    "\nSaved:"
)

print(feature_csv)
print(corr_csv)