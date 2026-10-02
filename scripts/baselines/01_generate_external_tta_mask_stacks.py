import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    r"C:\seg_uncertain\phase3_yolo_training"
    r"\YOLO_POLYP_SEG\weights\best.pt"
)

OUTPUT_ROOT = (
    r"C:\seg_uncertain\testing"
    r"\TTA_external"
)

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

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# VERIFY MODEL EXISTS
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model weights not found at:\n"
        f"  {MODEL_PATH}\n"
        f"Please check MODEL_PATH and ensure the weights file exists."
    )

# ============================================================
# LOAD MODEL
# (loaded once, reused for all datasets)
# ============================================================

print("\nLoading model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")

# ============================================================
# TTA AUGMENTATION FUNCTIONS
# Identical to 05A_part1_generate_mask_stacks.py
# ============================================================

def identity(img):
    return img


def hflip(img):
    return cv2.flip(img, 1)


def brightness_plus(img):

    out = img.astype(np.float32)

    out *= 1.10

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


def brightness_minus(img):

    out = img.astype(np.float32)

    out *= 0.90

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


def contrast_plus(img):

    alpha = 1.10

    out = cv2.convertScaleAbs(
        img,
        alpha=alpha,
        beta=0
    )

    return out


def contrast_minus(img):

    alpha = 0.90

    out = cv2.convertScaleAbs(
        img,
        alpha=alpha,
        beta=0
    )

    return out


def gamma_11(img):

    gamma = 1.1

    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(
        img,
        table
    )


def clahe_img(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    merged = cv2.merge([l, a, b])

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


def blur_img(img):

    return cv2.GaussianBlur(
        img,
        (5, 5),
        sigmaX=1
    )


# ============================================================
# TTA LIST
# Exactly 9 augmentations — identical to 05A_part1
# ============================================================

TTAS = [

    ("original",      identity),
    ("hflip",         hflip),
    ("bright_plus",   brightness_plus),
    ("bright_minus",  brightness_minus),
    ("contrast_plus", contrast_plus),
    ("contrast_minus",contrast_minus),
    ("gamma11",       gamma_11),
    ("clahe",         clahe_img),
    ("blur",          blur_img),

]

# ============================================================
# INFERENCE HELPER
# Identical logic to 05A_part1
# ============================================================

def get_prediction_mask(
    result,
    orig_h,
    orig_w,
    tta_name
):

    if result.masks is None:

        return np.zeros(
            (orig_h, orig_w),
            dtype=np.uint8
        )

    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )

    # --------------------------------
    # Merge all detected masks
    # --------------------------------

    merged = np.any(
        masks > 0,
        axis=0
    ).astype(np.uint8)

    # --------------------------------
    # Resize to original image size
    # --------------------------------

    merged = cv2.resize(

        merged,

        (orig_w, orig_h),

        interpolation=cv2.INTER_NEAREST

    )

    # --------------------------------
    # Undo horizontal flip
    # --------------------------------

    if tta_name == "hflip":

        merged = cv2.flip(
            merged,
            1
        )

    return merged

# ============================================================
# SANITY CHECKS
# ============================================================

def check_image_loaded(image, image_path):
    if image is None:
        raise ValueError(
            f"Failed to load image: {image_path}"
        )


def check_mask_shape(mask, expected_h, expected_w, image_name, tta_name):
    if mask.shape != (expected_h, expected_w):
        raise ValueError(
            f"Mask shape mismatch for {image_name} "
            f"(tta={tta_name}): "
            f"expected ({expected_h},{expected_w}), "
            f"got {mask.shape}"
        )


def check_stack_shape(stack, expected_n, expected_h, expected_w, image_name):
    if stack.shape != (expected_n, expected_h, expected_w):
        raise ValueError(
            f"Stack shape mismatch for {image_name}: "
            f"expected ({expected_n},{expected_h},{expected_w}), "
            f"got {stack.shape}"
        )


def check_consensus_shape(consensus, expected_h, expected_w, image_name):
    if consensus.shape != (expected_h, expected_w):
        raise ValueError(
            f"Consensus shape mismatch for {image_name}: "
            f"expected ({expected_h},{expected_w}), "
            f"got {consensus.shape}"
        )


def check_variance_range(variance, image_name):
    if variance.min() < 0.0 or variance.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Variance out of range [0,1] for {image_name}: "
            f"min={variance.min():.6f} max={variance.max():.6f}"
        )

# ============================================================
# MAIN PROCESSING LOOP — PER DATASET
# ============================================================

for dataset_name, info in DATASETS.items():

    image_dir = info["images"]

    # ── Output directories for this dataset ──────────────────

    dataset_out = os.path.join(
        OUTPUT_ROOT,
        dataset_name
    )

    mask_stack_dir = os.path.join(
        dataset_out,
        "mask_stacks"
    )

    consensus_dir = os.path.join(
        dataset_out,
        "consensus_masks"
    )

    variance_png_dir = os.path.join(
        dataset_out,
        "variance_maps_png"
    )

    variance_npy_dir = os.path.join(
        dataset_out,
        "variance_maps_npy"
    )

    for d in [
        dataset_out,
        mask_stack_dir,
        consensus_dir,
        variance_png_dir,
        variance_npy_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # ── Find images ──────────────────────────────────────────
    # Filter to supported image extensions only so that files
    # such as Thumbs.db, desktop.ini, .csv, .txt do not cause
    # cv2.imread to fail silently or raise an error.

    SUPPORTED_EXTENSIONS = (
        ".jpg", ".jpeg",
        ".png",
        ".bmp",
        ".tif", ".tiff"
    )

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower()
        in SUPPORTED_EXTENSIONS
    ])

    print("\n")
    print("=" * 60)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 60)
    print(f"Images: {len(image_files)}")
    print(f"TTA variants: {len(TTAS)}")

    log_rows = []

    # ── Per-image TTA loop ───────────────────────────────────

    for image_name in tqdm(
        image_files,
        desc=dataset_name
    ):

        image_path = os.path.join(
            image_dir,
            image_name
        )

        image = cv2.imread(image_path)

        check_image_loaded(image, image_path)

        h, w = image.shape[:2]

        stem = os.path.splitext(image_name)[0]

        mask_stack = []

        num_zero_preds = 0

        # ── Run all 9 TTA augmentations ─────────────────────

        for tta_name, tta_func in TTAS:

            tta_img = tta_func(image.copy())

            result = model(
                tta_img,
                verbose=False
            )[0]

            mask = get_prediction_mask(
                result,
                h,
                w,
                tta_name
            )

            check_mask_shape(
                mask, h, w,
                image_name, tta_name
            )

            if np.sum(mask) == 0:
                num_zero_preds += 1

            mask_stack.append(mask)

        # ── Stack ────────────────────────────────────────────

        mask_stack = np.stack(
            mask_stack,
            axis=0
        )  # shape: (9, H, W)

        check_stack_shape(
            mask_stack,
            len(TTAS), h, w,
            image_name
        )

        # ── Consensus ────────────────────────────────────────

        consensus = np.mean(
            mask_stack,
            axis=0
        )  # float64, values in [0,1]

        check_consensus_shape(
            consensus, h, w,
            image_name
        )

        # ── Variance ─────────────────────────────────────────

        variance = np.var(
            mask_stack,
            axis=0
        )

        # Normalize: max variance for binary var = 0.25 (p=0.5)
        variance = variance / 0.25

        check_variance_range(variance, image_name)

        # ── Save mask stack ──────────────────────────────────

        np.save(

            os.path.join(
                mask_stack_dir,
                stem + ".npy"
            ),

            mask_stack

        )

        # ── Save consensus PNG ───────────────────────────────

        consensus_img = (
            consensus * 255
        ).astype(np.uint8)

        cv2.imwrite(

            os.path.join(
                consensus_dir,
                stem + ".png"
            ),

            consensus_img

        )

        # ── Save consensus NPY (float64 for quantitative use) ─

        np.save(

            os.path.join(
                consensus_dir,
                stem + ".npy"
            ),

            consensus.astype(np.float32)

        )

        # ── Save variance PNG (visualization) ────────────────

        variance_img = (
            variance * 255
        ).astype(np.uint8)

        cv2.imwrite(

            os.path.join(
                variance_png_dir,
                stem + ".png"
            ),

            variance_img

        )

        # ── Save variance NPY (float32 for analysis) ─────────

        np.save(

            os.path.join(
                variance_npy_dir,
                stem + ".npy"
            ),

            variance.astype(np.float32)

        )

        # ── Log row ──────────────────────────────────────────

        log_rows.append({

            "image":
                image_name,

            "num_ttas":
                len(TTAS),

            "num_zero_predictions":
                num_zero_preds,

            "consensus_area":
                int(
                    (consensus > 0.5).sum()
                ),

            "dataset":
                dataset_name

        })

    # ── Save extraction log ──────────────────────────────────

    log_df = pd.DataFrame(log_rows)

    log_df.to_csv(

        os.path.join(
            dataset_out,
            "extraction_log.csv"
        ),

        index=False

    )

    # ── Save manifest ────────────────────────────────────────

    manifest = pd.DataFrame({
        "tta": [x[0] for x in TTAS]
    })

    manifest.to_csv(

        os.path.join(
            dataset_out,
            "tta_manifest.csv"
        ),

        index=False

    )

    # ── Dataset summary ──────────────────────────────────────

    avg_zero = log_df["num_zero_predictions"].mean()
    avg_consensus = log_df["consensus_area"].mean()
    n_failed = int(
        (log_df["num_zero_predictions"] == len(TTAS)).sum()
    )

    print(f"\nDataset:                    {dataset_name}")
    print(f"Images processed:           {len(log_rows)}")
    print(f"Avg zero predictions:       {avg_zero:.2f} / {len(TTAS)}")
    print(f"Avg consensus area (px):    {avg_consensus:.1f}")
    print(
        f"Images all-zero (failed):   {n_failed} "
        f"({100*n_failed/max(len(log_rows),1):.1f}%)"
    )
    print(f"Output directory:           {dataset_out}")

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("EXTERNAL TTA MASK STACK GENERATION COMPLETE")
print("=" * 60)

total = sum(
    len(pd.read_csv(
        os.path.join(OUTPUT_ROOT, ds, "extraction_log.csv")
    ))
    for ds in DATASETS
)

print(f"Total images processed: {total}")
print(f"TTA variants per image: {len(TTAS)}")
print(f"Output root:            {OUTPUT_ROOT}")