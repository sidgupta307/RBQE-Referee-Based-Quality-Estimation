import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import segmentation_models_pytorch as smp

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    r"C:\Medical_image_analysis\colon_cancer"
    r"\kvasir-seg\Kvasir-SEG"
    r"\best_unetplusplus_model.pth"
)

OUTPUT_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\23_unetpp_external_predictions_v2"
)

DATASETS = {
    "cvc_clinicdb":
        r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images",

    "cvc_colondb":
        r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images",

    "etis_larib":
        r"C:\Polygon_polyp_new(05_26)\etis-larib\images",

    "cvc_300":
        r"C:\Polygon_polyp_new(05_26)\CVC-300\images"
}

# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 512

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# MODEL
# ============================================================

print("\n")
print("=" * 60)
print("UNET++ EXTERNAL PREDICTIONS V2")
print("=" * 60)
print("DEVICE:", DEVICE)

print("\nLoading UNet++...")

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("UNet++ Loaded.")

# ============================================================
# INFERENCE
# ============================================================

total_images = 0

with torch.no_grad():

    for dataset_name, image_dir in DATASETS.items():

        print("\n")
        print("=" * 60)
        print(dataset_name.upper())
        print("=" * 60)

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

        print("Images:", len(image_files))

        save_mask_dir = os.path.join(
            OUTPUT_ROOT,
            dataset_name,
            "masks"
        )

        os.makedirs(
            save_mask_dir,
            exist_ok=True
        )

        metadata = []

        for image_name in tqdm(image_files):

            image_path = os.path.join(
                image_dir,
                image_name
            )

            image_bgr = cv2.imread(
                image_path
            )

            if image_bgr is None:
                continue

            original_h, original_w = (
                image_bgr.shape[:2]
            )

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            # =====================================
            # TRAINING PREPROCESSING
            # =====================================

            image_resized = cv2.resize(
                image_rgb,
                (INPUT_SIZE, INPUT_SIZE)
            )

            image_resized = (
                image_resized.astype(np.float32)
                / 255.0
            )

            image_resized = (
                image_resized
                - IMAGENET_MEAN
            ) / IMAGENET_STD

            image_tensor = torch.tensor(
                image_resized
            ).permute(
                2, 0, 1
            ).unsqueeze(
                0
            ).float().to(
                DEVICE
            )

            # =====================================
            # INFERENCE
            # =====================================

            logits = model(
                image_tensor
            )

            probs = torch.sigmoid(
                logits
            )

            pred = (
                probs > 0.5
            ).float()

            pred = (
                pred.squeeze()
                .cpu()
                .numpy()
            )

            pred = cv2.resize(
                pred,
                (
                    original_w,
                    original_h
                ),
                interpolation=cv2.INTER_NEAREST
            )

            mask = (
                pred.astype(np.uint8)
                * 255
            )

            save_name = (
                os.path.splitext(
                    image_name
                )[0]
                + ".png"
            )

            save_path = os.path.join(
                save_mask_dir,
                save_name
            )

            cv2.imwrite(
                save_path,
                mask
            )

            metadata.append([
                image_name,
                int(pred.sum())
            ])

            total_images += 1

        metadata_df = pd.DataFrame(
            metadata,
            columns=[
                "image",
                "pred_area"
            ]
        )

        metadata_df.to_csv(
            os.path.join(
                OUTPUT_ROOT,
                dataset_name,
                "metadata.csv"
            ),
            index=False
        )

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("UNET++ EXTERNAL PREDICTIONS COMPLETE")
print("=" * 60)

print(
    "Images Processed:",
    total_images
)

print(
    "\nSaved To:"
)

print(
    OUTPUT_ROOT
)