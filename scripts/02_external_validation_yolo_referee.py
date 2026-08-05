import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

from ultralytics import YOLO

import torch
import torch.nn.functional as F

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee"
    r"\02_external_validation"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

# ============================================================
# MODELS
# ============================================================

YOLO_MODEL_PATH = (
    r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee"
    r"\YOLO_REFEREE_SEED123\weights\best.pt"
)

SEGFORMER_MODEL_PATH = (
    r"C:\seg_uncertain\segformer"
    r"\checkpoints\best_model.pth"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 512

# ============================================================
# DATASETS
# ============================================================

DATASETS = {

    "cvc_clinicdb": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images",

        "masks":
            r"C:\Polygon_polyp_new(05_26)\cvc_dataset\masks"

    },

    "cvc_colondb": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images",

        "masks":
            r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\masks"

    },

    "etis_larib": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\etis-larib\images",

        "masks":
            r"C:\Polygon_polyp_new(05_26)\etis-larib\masks"

    },

    "cvc_300": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\CVC-300\images",

        "masks":
            r"C:\Polygon_polyp_new(05_26)\CVC-300\masks"

    }

}

# ============================================================
# METRICS
# ============================================================

def dice_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    return (
        2 * inter + 1e-8
    ) / (
        pred.sum()
        + gt.sum()
        + 1e-8
    )


def iou_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    return (
        inter + 1e-8
    ) / (
        union + 1e-8
    )


def precision_score(pred, gt):

    tp = np.logical_and(
        pred == 1,
        gt == 1
    ).sum()

    fp = np.logical_and(
        pred == 1,
        gt == 0
    ).sum()

    return tp / (tp + fp + 1e-8)


def recall_score(pred, gt):

    tp = np.logical_and(
        pred == 1,
        gt == 1
    ).sum()

    fn = np.logical_and(
        pred == 0,
        gt == 1
    ).sum()

    return tp / (tp + fn + 1e-8)

# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

print("YOLO Loaded.")

# ============================================================
# LOAD SEGFORMER
# ============================================================

print("\nLoading SegFormer...")

processor = SegformerImageProcessor(
    do_resize=False,
    do_normalize=True,
    do_rescale=True
)

segformer = (
    SegformerForSemanticSegmentation
    .from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
)

segformer.load_state_dict(

    torch.load(
        SEGFORMER_MODEL_PATH,
        map_location=DEVICE
    )

)

segformer = segformer.to(
    DEVICE
)

segformer.eval()

print("SegFormer Loaded.")

# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

def predict_yolo(
    image_path,
    h,
    w
):

    result = yolo_model(
        image_path,
        verbose=False
    )[0]

    pred_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if result.masks is not None:

        mask = (
            result.masks
            .data[0]
            .cpu()
            .numpy()
        )

        pred_mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        pred_mask = (
            pred_mask > 0.5
        ).astype(np.uint8)

    return pred_mask


def predict_segformer(
    image
):

    h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    encoded = processor(
        images=image_resized,
        return_tensors="pt"
    )

    pixel_values = encoded[
        "pixel_values"
    ].to(DEVICE)

    with torch.no_grad():

        outputs = segformer(
            pixel_values=pixel_values
        )

        logits = F.interpolate(

            outputs.logits,

            size=(h, w),

            mode="bilinear",

            align_corners=False

        )

        pred = torch.argmax(
            logits,
            dim=1
        )

    pred = (
        pred
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return pred

# ============================================================
# MAIN LOOP
# ============================================================

all_rows = []

for dataset_name, paths in DATASETS.items():

    print("\n")
    print("=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    image_dir = paths["images"]
    mask_dir = paths["masks"]

    dataset_rows = []

    image_files = sorted(
        os.listdir(image_dir)
    )

    for image_name in tqdm(image_files):

        stem = os.path.splitext(
            image_name
        )[0]

        image_path = os.path.join(
            image_dir,
            image_name
        )

        gt_path = os.path.join(
            mask_dir,
            stem + ".png"
        )

        if not os.path.exists(gt_path):
            continue

        image = cv2.imread(
            image_path
        )

        gt = cv2.imread(
            gt_path,
            0
        )

        h, w = image.shape[:2]

        gt = cv2.resize(
            gt,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        gt = (
            gt > 127
        ).astype(np.uint8)

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        yolo_mask = predict_yolo(
            image_path,
            h,
            w
        )

        # ----------------------------------------------------
        # SEGFORMER
        # ----------------------------------------------------

        seg_mask = predict_segformer(
            image
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        for model_name, pred in [

            ("YOLOv8-Seg", yolo_mask),

            ("SegFormer-B0", seg_mask)

        ]:

            row = {

                "dataset":
                    dataset_name,

                "image":
                    image_name,

                "model":
                    model_name,

                "dice":
                    dice_score(
                        pred,
                        gt
                    ),

                "iou":
                    iou_score(
                        pred,
                        gt
                    ),

                "precision":
                    precision_score(
                        pred,
                        gt
                    ),

                "recall":
                    recall_score(
                        pred,
                        gt
                    )

            }

            dataset_rows.append(
                row
            )

            all_rows.append(
                row
            )

    # =======================================================
    # SAVE DATASET
    # =======================================================

    dataset_df = pd.DataFrame(
        dataset_rows
    )

    dataset_df.to_csv(

        os.path.join(
            OUTPUT_ROOT,
            f"{dataset_name}_per_image.csv"
        ),

        index=False
    )

    summary = (

        dataset_df
        .groupby("model")
        [["dice",
          "iou",
          "precision",
          "recall"]]
        .mean()
        .reset_index()

    )

    summary.to_csv(

        os.path.join(
            OUTPUT_ROOT,
            f"{dataset_name}_summary.csv"
        ),

        index=False
    )

# ============================================================
# SAVE COMBINED
# ============================================================

all_df = pd.DataFrame(
    all_rows
)

all_df.to_csv(

    os.path.join(
        OUTPUT_ROOT,
        "combined_results.csv"
    ),

    index=False
)

# ============================================================
# PAPER TABLE
# ============================================================

paper_table = (

    all_df
    .groupby(
        ["dataset", "model"]
    )
    [
        [
            "dice",
            "iou",
            "precision",
            "recall"
        ]
    ]
    .mean()
    .reset_index()

)

paper_table.to_csv(

    os.path.join(
        OUTPUT_ROOT,
        "paper_table_external_validation.csv"
    ),

    index=False
)

print("\n")
print("=" * 60)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 60)

print(
    f"Results Saved:\n{OUTPUT_ROOT}"
)