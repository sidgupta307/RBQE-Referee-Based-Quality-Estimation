import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import transform
import torch.nn.functional as F

from segment_anything import sam_model_registry

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
# YOLO PREDICTIONS
# ============================================================

YOLO_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\02_external_predictions"
)

# ============================================================
# MEDSAM
# ============================================================

CHECKPOINT = (
    r"C:\MedSAM\work_dir\MedSAM"
    r"\medsam_vit_b.pth"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    r"C:\seg_uncertain\chat_gpt"
    r"\04_medsam_external_masks"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n====================================")
print("EXTERNAL MEDSAM GENERATION")
print("====================================")
print("DEVICE:", DEVICE)

# ============================================================
# LOAD MEDSAM
# ============================================================

print("\nLoading MedSAM...")

medsam_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

medsam_model = medsam_model.to(
    DEVICE
)

medsam_model.eval()

print("MedSAM Loaded.")

# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def medsam_inference(
    medsam_model,
    img_embed,
    box_1024,
    H,
    W
):

    box_torch = torch.as_tensor(
        box_1024,
        dtype=torch.float,
        device=img_embed.device
    )

    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = (
        medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
    )

    low_res_logits, _ = (

        medsam_model.mask_decoder(

            image_embeddings=img_embed,

            image_pe=
            medsam_model.prompt_encoder.get_dense_pe(),

            sparse_prompt_embeddings=
            sparse_embeddings,

            dense_prompt_embeddings=
            dense_embeddings,

            multimask_output=False

        )

    )

    low_res_pred = torch.sigmoid(
        low_res_logits
    )

    low_res_pred = F.interpolate(

        low_res_pred,

        size=(H, W),

        mode="bilinear",

        align_corners=False

    )

    low_res_pred = (
        low_res_pred
        .squeeze()
        .cpu()
        .numpy()
    )

    medsam_seg = (
        low_res_pred > 0.5
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            medsam_seg,
            connectivity=8
        )
    )

    if num_labels > 1:

        largest_component = (

            1 +
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )

        )

        medsam_seg = (
            labels == largest_component
        ).astype(np.uint8)

    return medsam_seg

# ============================================================
# HELPERS
# ============================================================

def mask_to_bbox(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    return np.array(

        [

            xs.min(),
            ys.min(),

            xs.max(),
            ys.max()

        ],

        dtype=np.float32

    )

def create_embedding(
    image_rgb
):

    img_1024 = transform.resize(

        image_rgb,

        (1024, 1024),

        order=3,

        preserve_range=True,

        anti_aliasing=True

    ).astype(np.uint8)

    img_1024 = (

        img_1024
        -
        img_1024.min()

    ) / np.clip(

        img_1024.max()
        -
        img_1024.min(),

        a_min=1e-8,

        a_max=None

    )

    img_tensor = (

        torch.tensor(
            img_1024
        )

        .float()

        .permute(2,0,1)

        .unsqueeze(0)

        .to(DEVICE)

    )

    with torch.no_grad():

        image_embedding = (
            medsam_model.image_encoder(
                img_tensor
            )
        )

    return image_embedding

# ============================================================
# MAIN
# ============================================================

combined_metadata = []

for dataset_name, info in DATASETS.items():

    print("\n")
    print("=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    image_dir = info["images"]

    yolo_mask_dir = os.path.join(

        YOLO_ROOT,
        dataset_name,
        "masks"

    )

    dataset_out = os.path.join(
        OUTPUT_ROOT,
        dataset_name
    )

    mask_out_dir = os.path.join(
        dataset_out,
        "masks"
    )

    os.makedirs(
        mask_out_dir,
        exist_ok=True
    )

    metadata = []

    image_files = sorted(
        os.listdir(image_dir)
    )

    print(
        f"Images: {len(image_files)}"
    )

    for image_name in tqdm(image_files):

        stem = os.path.splitext(
            image_name
        )[0]

        image_path = os.path.join(
            image_dir,
            image_name
        )

        yolo_mask_path = os.path.join(
            yolo_mask_dir,
            stem + ".png"
        )

        save_mask_path = os.path.join(
            mask_out_dir,
            stem + ".png"
        )

        image_bgr = cv2.imread(
            image_path
        )

        H, W = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        yolo_mask = cv2.imread(
            yolo_mask_path,
            0
        )

        if yolo_mask is None:
            continue

        bbox = mask_to_bbox(
            yolo_mask
        )

        if bbox is None:

            empty_mask = np.zeros(
                (H, W),
                dtype=np.uint8
            )

            cv2.imwrite(
                save_mask_path,
                empty_mask
            )

            row = {

                "dataset":
                    dataset_name,

                "image":
                    image_name,

                "medsam_run":
                    0,

                "yolo_area":
                    0,

                "medsam_area":
                    0

            }

            metadata.append(row)
            combined_metadata.append(row)

            continue

        image_embedding = create_embedding(
            image_rgb
        )

        box_1024 = (

            bbox

            /

            np.array(
                [W,H,W,H]
            )

        ) * 1024

        box_1024 = box_1024[None,:]

        medsam_mask = medsam_inference(

            medsam_model,

            image_embedding,

            box_1024,

            H,

            W

        )

        cv2.imwrite(

            save_mask_path,

            (
                medsam_mask * 255
            ).astype(np.uint8)

        )

        row = {

            "dataset":
                dataset_name,

            "image":
                image_name,

            "medsam_run":
                1,

            "yolo_area":
                int(
                    np.sum(
                        yolo_mask > 0
                    )
                ),

            "medsam_area":
                int(
                    np.sum(
                        medsam_mask > 0
                    )
                ),

            "bbox_x1":
                int(bbox[0]),

            "bbox_y1":
                int(bbox[1]),

            "bbox_x2":
                int(bbox[2]),

            "bbox_y2":
                int(bbox[3])

        }

        metadata.append(row)
        combined_metadata.append(row)

    pd.DataFrame(
        metadata
    ).to_csv(

        os.path.join(
            dataset_out,
            "metadata.csv"
        ),

        index=False

    )

# ============================================================
# SAVE MASTER CSV
# ============================================================

pd.DataFrame(
    combined_metadata
).to_csv(

    os.path.join(
        OUTPUT_ROOT,
        "combined_metadata.csv"
    ),

    index=False

)

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("EXTERNAL MEDSAM COMPLETE")
print("=" * 60)

print(
    f"Images Processed: "
    f"{len(combined_metadata)}"
)

print(
    f"Saved To:\n{OUTPUT_ROOT}"
)