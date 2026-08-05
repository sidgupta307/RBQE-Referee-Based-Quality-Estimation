import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# MODEL
# ============================================================

YOLO_MODEL_PATH = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\YOLO_REFEREE_SEED123"
    r"\weights\best.pt"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    r"C:\seg_uncertain\journal_extension"
    r"\T1_22_Independent_YOLO_Referee"
    r"\03_external_referee_predictions"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

# ============================================================
# DATASETS
# ============================================================

DATASETS = {

    "cvc_clinicdb": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images"

    },

    "cvc_colondb": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images"

    },

    "etis_larib": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\etis-larib\images"

    },

    "cvc_300": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\CVC-300\images"

    }

}

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO...")

model = YOLO(
    YOLO_MODEL_PATH
)

print("YOLO Loaded.")

# ============================================================
# MAIN
# ============================================================

for dataset_name, info in DATASETS.items():

    print("\n")
    print("=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    image_dir = info["images"]

    dataset_root = os.path.join(
        OUTPUT_ROOT,
        dataset_name
    )

    mask_dir = os.path.join(
        dataset_root,
        "masks"
    )

    prob_dir = os.path.join(
        dataset_root,
        "probabilities"
    )

    os.makedirs(
        mask_dir,
        exist_ok=True
    )

    os.makedirs(
        prob_dir,
        exist_ok=True
    )

    rows = []

    image_files = sorted(
        os.listdir(image_dir)
    )

    print(
        f"Images: {len(image_files)}"
    )

    for image_name in tqdm(image_files):

        image_path = os.path.join(
            image_dir,
            image_name
        )

        stem = os.path.splitext(
            image_name
        )[0]

        image = cv2.imread(
            image_path
        )

        h, w = image.shape[:2]

        result = model(
            image_path,
            verbose=False
        )[0]

        pred_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        prob_map = np.zeros(
            (h, w),
            dtype=np.float32
        )

        confidence = 0.0

        detected = 0

        if result.masks is not None:

            raw_prob = (
                result.masks
                .data[0]
                .cpu()
                .numpy()
            )

            prob_map = cv2.resize(

                raw_prob,

                (w, h),

                interpolation=cv2.INTER_LINEAR

            )

            pred_mask = (
                prob_map > 0.5
            ).astype(np.uint8)

            detected = 1

            if result.boxes is not None:

                confidence = float(
                    result.boxes.conf[0]
                )

        # ====================================================
        # SAVE MASK
        # ====================================================

        cv2.imwrite(

            os.path.join(
                mask_dir,
                stem + ".png"
            ),

            pred_mask * 255

        )

        # ====================================================
        # SAVE PROBABILITY MAP
        # ====================================================

        np.save(

            os.path.join(
                prob_dir,
                stem + ".npy"
            ),

            prob_map.astype(
                np.float32
            )

        )

        rows.append({

            "image":
                image_name,

            "detected":
                detected,

            "confidence":
                confidence,

            "mask_area":
                int(
                    pred_mask.sum()
                )

        })

    metadata = pd.DataFrame(
        rows
    )

    metadata.to_csv(

        os.path.join(
            dataset_root,
            "metadata.csv"
        ),

        index=False
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("EXTERNAL YOLO PREDICTIONS COMPLETE")
print("=" * 60)

print(
    f"Saved To:\n{OUTPUT_ROOT}"
)