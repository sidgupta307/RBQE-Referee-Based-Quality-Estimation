import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

AGREEMENT_CSV = (
    r"C:\seg_uncertain\chat_gpt"
    r"\20_segformer_agreement\agreement_features.csv"
)

YOLO_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\02_external_predictions"
)

SEG_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\19_segformer_external_predictions"
)

OUTPUT_DIR = (
    r"C:\seg_uncertain\chat_gpt"
    r"\20B_segformer_clean_agreement"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    AGREEMENT_CSV
)

print("Original Images:",
      len(df))

keep_rows = []

both_empty = 0

# ============================================================
# FILTER
# ============================================================

for _, row in df.iterrows():

    dataset = row["dataset"]
    image = row["image"]

    mask_name = (
        os.path.splitext(image)[0]
        + ".png"
    )

    yolo_mask = cv2.imread(
        os.path.join(
            YOLO_ROOT,
            dataset,
            "masks",
            mask_name
        ),
        0
    )

    seg_mask = cv2.imread(
        os.path.join(
            SEG_ROOT,
            dataset,
            "masks",
            mask_name
        ),
        0
    )

    yolo_area = np.sum(
        yolo_mask > 0
    )

    seg_area = np.sum(
        seg_mask > 0
    )

    if (
        yolo_area == 0
        and
        seg_area == 0
    ):
        both_empty += 1
        continue

    keep_rows.append(row)

# ============================================================
# SAVE
# ============================================================

clean_df = pd.DataFrame(
    keep_rows
)

clean_csv = os.path.join(
    OUTPUT_DIR,
    "clean_agreement_features.csv"
)

clean_df.to_csv(
    clean_csv,
    index=False
)

print("\nRemoved both-empty:",
      both_empty)

print(
    "Remaining Images:",
    len(clean_df)
)

print("\nSaved:")
print(clean_csv)