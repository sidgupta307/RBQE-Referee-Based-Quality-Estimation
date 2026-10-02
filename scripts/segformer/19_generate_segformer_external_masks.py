import os
import cv2
import numpy as np

from tqdm import tqdm

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
    r"C:\seg_uncertain\chat_gpt"
    r"\19_segformer_external_predictions"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
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

        "image_ext":
            ".jpg"

    },

    "cvc_colondb": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images",

        "image_ext":
            ".png"

    },

    "etis_larib": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\etis-larib\images",

        "image_ext":
            ".png"

    },

    "cvc_300": {

        "images":
            r"C:\Polygon_polyp_new(05_26)\CVC-300\images",

        "image_ext":
            ".png"

    }

}

# ============================================================
# PROCESSOR
# ============================================================

processor = SegformerImageProcessor(
    do_resize=False,
    do_normalize=True,
    do_rescale=True
)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n====================================")
print("SEGFORMER EXTERNAL PREDICTIONS")
print("====================================")
print("DEVICE:", DEVICE)

print("\nLoading SegFormer...")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

)

model = model.to(DEVICE)

model.eval()

print("SegFormer Loaded.")

# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_segformer(image_bgr):

    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(
        image_bgr,
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

    outputs = model(
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

total_images = 0

for dataset_name, cfg in DATASETS.items():

    print("\n")
    print("=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    image_dir = cfg["images"]

    save_dir = os.path.join(
        OUTPUT_ROOT,
        dataset_name,
        "masks"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    image_files = sorted(

        [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                cfg["image_ext"]
            )
        ]

    )

    print(
        "Images:",
        len(image_files)
    )

    total_images += len(image_files)

    for image_name in tqdm(image_files):

        image_path = os.path.join(
            image_dir,
            image_name
        )

        image = cv2.imread(
            image_path
        )

        pred = predict_segformer(
            image
        )

        pred_mask = (
            pred * 255
        ).astype(np.uint8)

        stem = os.path.splitext(
            image_name
        )[0]

        save_path = os.path.join(
            save_dir,
            stem + ".png"
        )

        cv2.imwrite(
            save_path,
            pred_mask
        )

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("SEGFORMER EXTERNAL PREDICTIONS COMPLETE")
print("=" * 60)

print(
    "Images Processed:",
    total_images
)

print(
    "Saved To:"
)

print(
    OUTPUT_ROOT
)