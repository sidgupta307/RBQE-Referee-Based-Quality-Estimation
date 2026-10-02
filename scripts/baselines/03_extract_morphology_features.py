import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import spearmanr

# ============================================================
# INPUTS
# ============================================================

PRED_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\02_external_predictions"
)

METRICS_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\01_external_validation"
    r"\combined_results.csv"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\03_morphology_features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD METRICS
# ============================================================

metrics_df = pd.read_csv(
    METRICS_CSV
)

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# We only use YOLO rows
# because all later reliability
# analysis will be based on YOLO.
# ------------------------------------------------------------

metrics_df = metrics_df[
    metrics_df["model"] == "YOLOv8-Seg"
].copy()

print(
    f"\nLoaded YOLO rows: "
    f"{len(metrics_df)}"
)

# ============================================================
# HELPERS
# ============================================================

def largest_contour(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    return max(
        contours,
        key=cv2.contourArea
    )


def morphology_features(mask):

    contour = largest_contour(mask)

    area = float(
        np.sum(mask > 0)
    )

    if contour is None:

        return {

            "area": area,
            "perimeter": 0.0,
            "circularity": 0.0,
            "compactness": 0.0,
            "solidity": 0.0,
            "aspect_ratio": 0.0,
            "boundary_complexity": 0.0

        }

    perimeter = float(
        cv2.arcLength(
            contour,
            True
        )
    )

    # --------------------------------------------------------
    # Circularity
    # --------------------------------------------------------

    if perimeter > 0:

        circularity = (
            4.0
            * np.pi
            * area
        ) / (
            perimeter**2
            + 1e-8
        )

        compactness = (
            perimeter**2
        ) / (
            4.0
            * np.pi
            * area
            + 1e-8
        )

    else:

        circularity = 0.0
        compactness = 0.0

    # --------------------------------------------------------
    # Solidity
    # --------------------------------------------------------

    hull = cv2.convexHull(
        contour
    )

    hull_area = float(
        cv2.contourArea(
            hull
        )
    )

    if hull_area > 0:

        solidity = (
            area
            /
            hull_area
        )

    else:

        solidity = 0.0

    # --------------------------------------------------------
    # Aspect Ratio
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        contour
    )

    aspect_ratio = (
        w
        /
        (h + 1e-8)
    )

    # --------------------------------------------------------
    # Boundary Complexity
    # --------------------------------------------------------

    boundary_complexity = (
        perimeter
        /
        (
            np.sqrt(area)
            + 1e-8
        )
    )

    return {

        "area":
            area,

        "perimeter":
            perimeter,

        "circularity":
            circularity,

        "compactness":
            compactness,

        "solidity":
            solidity,

        "aspect_ratio":
            aspect_ratio,

        "boundary_complexity":
            boundary_complexity

    }

# ============================================================
# FEATURE EXTRACTION
# ============================================================

rows = []

for _, row in tqdm(
    metrics_df.iterrows(),
    total=len(metrics_df)
):

    dataset = row["dataset"]

    image_name = row["image"]

    stem = os.path.splitext(
        image_name
    )[0]

    mask_path = os.path.join(

        PRED_ROOT,
        dataset,
        "masks",
        stem + ".png"

    )

    if not os.path.exists(
        mask_path
    ):
        continue

    mask = cv2.imread(
        mask_path,
        0
    )

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    feats = morphology_features(
        mask
    )

    feats.update({

        "dataset":
            dataset,

        "image":
            image_name,

        "dice":
            row["dice"],

        "iou":
            row["iou"],

        "precision":
            row["precision"],

        "recall":
            row["recall"]

    })

    rows.append(
        feats
    )

# ============================================================
# SAVE FEATURES
# ============================================================

features_df = pd.DataFrame(
    rows
)

feature_csv = os.path.join(
    OUTPUT_DIR,
    "morphology_features.csv"
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
# CORRELATION ANALYSIS
# ============================================================

feature_cols = [

    "area",
    "perimeter",
    "circularity",
    "compactness",
    "solidity",
    "aspect_ratio",
    "boundary_complexity"

]

ranking = []

for feat in feature_cols:

    rho, p = spearmanr(

        features_df[feat],
        features_df["dice"]

    )

    ranking.append({

        "feature":
            feat,

        "spearman_rho":
            rho,

        "p_value":
            p

    })

ranking_df = (

    pd.DataFrame(
        ranking
    )

    .sort_values(
        "spearman_rho",
        ascending=False
    )

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
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MORPHOLOGY FEATURE RANKING")
print("=" * 70)

print(
    ranking_df
)

print("\n")
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Images: {len(features_df)}"
)
print(
    f"Features: {len(feature_cols)}"
)