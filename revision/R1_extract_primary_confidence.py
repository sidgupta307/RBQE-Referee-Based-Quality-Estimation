#!/usr/bin/env python3
"""
Reviewer 6 (Comment 2) and Reviewer 5 (Comment 1):
extract the primary YOLOv8n-Seg model's own confidence for every external image.

Two label-free confidence scores are recorded per image (fixed before looking
at any result):
  conf_det : confidence of the top reported detection at the deployed setting
             (Ultralytics default conf = 0.25). This is the detection whose mask
             is used as the primary mask (masks.data[0]). Empty prediction -> 0.
  conf_max : highest box confidence of any candidate when the model is run with
             conf = 0.001. Gives a continuous score even when the deployed
             prediction is empty. Used only as a score; the primary mask is
             still the deployed (conf = 0.25) mask.
For alignment, the script also recomputes the primary mask's DSC against the
ground truth exactly as in scripts/primary/01_external_validation_yolo_segformer.py.

Usage (Windows example):
  python R1_extract_primary_confidence.py ^
      --weights "C:\\seg_uncertain\\phase3_yolo_training\\YOLO_POLYP_SEG\\weights\\best.pt" ^
      --out primary_confidence.csv --gpu 1
Dataset folders default to the paths used in the repository scripts; change
DATASETS below if yours differ.

GPU choice: --gpu selects the GPU by the index shown in nvidia-smi (default 1,
the idle GPU on the lab machine). Only that GPU is made visible to PyTorch, so
the job cannot touch the other one.
"""
import argparse
import os
import sys

# ---- pick the GPU before torch/ultralytics are imported -------------------
_gpu = "1"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
    elif arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # same numbering as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

DATASETS = {
    "cvc_clinicdb": (r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images",
                     r"C:\Polygon_polyp_new(05_26)\cvc_dataset\masks"),
    "cvc_colondb": (r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images",
                    r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\masks"),
    "etis_larib": (r"C:\Polygon_polyp_new(05_26)\etis-larib\images",
                   r"C:\Polygon_polyp_new(05_26)\etis-larib\masks"),
    "cvc_300": (r"C:\Polygon_polyp_new(05_26)\CVC-300\images",
                r"C:\Polygon_polyp_new(05_26)\CVC-300\masks"),
}

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True, help="primary YOLOv8n-Seg best.pt (seed 42)")
p.add_argument("--out", default="primary_confidence.csv")
p.add_argument("--gpu", default="1", help="GPU index as shown by nvidia-smi (default 1)")
a = p.parse_args()

import torch
if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")
DEV = 0                     # the only visible device is the chosen GPU
model = YOLO(a.weights)


def dice(x, y):
    s = x.sum() + y.sum()
    return 2.0 * np.logical_and(x, y).sum() / s if s else 0.0


rows = []
for ds, (img_dir, msk_dir) in DATASETS.items():
    for name in tqdm(sorted(os.listdir(img_dir)), desc=ds):
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(img_dir, name)
        gt_path = os.path.join(msk_dir, stem + ".png")
        if not os.path.exists(gt_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt = cv2.imread(gt_path, 0)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST) > 127

        r = model(img_path, verbose=False, device=DEV)[0]                 # deployed setting
        n_det = 0 if r.boxes is None else len(r.boxes)
        conf_det = float(r.boxes.conf[0]) if n_det else 0.0
        mask = np.zeros((h, w), bool)
        if r.masks is not None:
            mk = r.masks.data[0].cpu().numpy()
            mask = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5

        r2 = model(img_path, verbose=False, conf=0.001, device=DEV)[0]    # all candidates
        conf_max = float(r2.boxes.conf.max()) if (r2.boxes is not None and len(r2.boxes)) else 0.0

        rows.append((ds, stem, n_det, conf_det, conf_max, int(mask.sum()), dice(mask, gt)))

import csv
with open(a.out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["dataset", "image", "n_det", "conf_det", "conf_max", "mask_pixels", "dsc_check"])
    wr.writerows(rows)
print("Saved", len(rows), "rows to", os.path.abspath(a.out))
print("Images with empty primary prediction:", sum(1 for r in rows if r[2] == 0))
