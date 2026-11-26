#!/usr/bin/env python3
"""
Reviewer 6, Comment 6 - TTA baseline on the PolypDB external set.

Runs the TTA baseline exactly as on the benchmark
(scripts/baselines/01_generate_external_tta_mask_stacks.py and 02_...):
nine inference conditions on the primary YOLOv8n-Seg model (original, horizontal
flip, brightness x1.10 / x0.90, contrast x1.10 / x0.90, gamma 1.1, CLAHE,
Gaussian blur 5x5), all detected masks merged, and the TTA score is the number
of conditions that give an empty mask (num_zero_predictions, 0-9).
Same PolypDB folders, groups and mask conventions as R7B_polypdb_external.py.

Usage (Windows, from C:\\seg_uncertain):
  python R10_tta_polypdb.py --polypdb "C:\\PolypDB\\PolypDB\\PolypDB" --gpu 1
Output: polypdb_tta.csv
"""
import argparse
import os
import sys

_gpu = "1"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--polypdb", required=True, help="folder that contains PolypDB_center_wise")
p.add_argument("--gpu", default="1")
p.add_argument("--primary", default=r"C:\seg_uncertain\phase3_yolo_training\YOLO_POLYP_SEG\weights\best.pt")
p.add_argument("--out", default="polypdb_tta.csv")
a = p.parse_args()
if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")

JOBS = [("main", "BKAI/WLI"), ("main", "BKAI/BLI"), ("main", "BKAI/FICE"), ("main", "BKAI/LCI"),
        ("main", "Karolinska/WLI"), ("sensitivity", "Simula/NBI")]
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
model = YOLO(a.primary)

GAMMA = np.array([((i / 255.0) ** (1.0 / 1.1)) * 255 for i in range(256)]).astype("uint8")


def clahe(img):
    l, aa, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, aa, b]), cv2.COLOR_LAB2BGR)


TTAS = [lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: np.clip(x.astype(np.float32) * 1.10, 0, 255).astype(np.uint8),
        lambda x: np.clip(x.astype(np.float32) * 0.90, 0, 255).astype(np.uint8),
        lambda x: cv2.convertScaleAbs(x, alpha=1.10, beta=0),
        lambda x: cv2.convertScaleAbs(x, alpha=0.90, beta=0),
        lambda x: cv2.LUT(x, GAMMA),
        clahe,
        lambda x: cv2.GaussianBlur(x, (5, 5), sigmaX=1)]


def sub(root, names):
    for n in names:
        if os.path.isdir(os.path.join(root, n)):
            return os.path.join(root, n)
    return None


rows = []
for group, cm in JOBS:
    centre, modality = cm.split("/")
    root = os.path.join(a.polypdb, "PolypDB_center_wise", centre, modality)
    idir = sub(root, ["images", "Images"]); mdir = sub(root, ["masks", "annotations", "Masks", "Annotations"])
    if idir is None or mdir is None:
        print("WARNING: folder not found:", root); continue
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(mdir) if f.lower().endswith(EXT)}
    for f in tqdm(sorted(x for x in os.listdir(idir) if x.lower().endswith(EXT)), desc=cm):
        stem = os.path.splitext(f)[0]
        if stem not in masks:
            continue
        img = cv2.imread(os.path.join(idir, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        gm = cv2.resize(cv2.imread(os.path.join(mdir, masks[stem]), cv2.IMREAD_COLOR), (w, h), interpolation=cv2.INTER_NEAREST)
        if not (gm.max(axis=2) > 127).any():
            continue
        zero = 0
        for t in TTAS:
            r = model(t(img.copy()), verbose=False, device=0)[0]
            empty = r.masks is None or not np.any(r.masks.data.cpu().numpy() > 0)
            zero += int(empty)
        rows.append([group, centre, modality, stem, zero])
with open(a.out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["group", "centre", "modality", "image", "num_zero_predictions"])
    wr.writerows(rows)
print("Saved", len(rows), "rows to", os.path.abspath(a.out))
