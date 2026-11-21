#!/usr/bin/env python3
"""
Reviewer 3, Comment 1 - evaluate the 512 x 512 Independent YOLO Referee
(R3A_yolo_referee_512.py) on the PolypDB external set, with the same folders,
groups and mask conventions as R7B_polypdb_external.py.

Usage (Windows, from C:\\seg_uncertain):
  python R3B_yolo512_polypdb.py --polypdb "C:\\PolypDB\\PolypDB\\PolypDB" --gpu 1
Output: polypdb_yolo512.csv
"""
import argparse
import os
import sys

_gpu = "1"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
    elif arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

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
p.add_argument("--yolo512", default=r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee\YOLO_REFEREE_SEED123_IMG512\weights\best.pt")
p.add_argument("--out", default="polypdb_yolo512.csv")
a = p.parse_args()

if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")
JOBS = [("main", "BKAI/WLI"), ("main", "BKAI/BLI"), ("main", "BKAI/FICE"), ("main", "BKAI/LCI"),
        ("main", "Karolinska/WLI"), ("sensitivity", "Simula/NBI")]
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
primary = YOLO(a.primary); ref = YOLO(a.yolo512)


def dice(x, y):
    s = x.sum() + y.sum(); return 2.0 * np.logical_and(x, y).sum() / s if s else 0.0


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
        ip = os.path.join(idir, f); img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gm = cv2.resize(cv2.imread(os.path.join(mdir, masks[stem]), cv2.IMREAD_COLOR), (w, h), interpolation=cv2.INTER_NEAREST)
        gt = gm.max(axis=2) > 127
        if not gt.any():
            continue
        r = primary(ip, verbose=False, device=0)[0]
        sp = (cv2.resize(r.masks.data[0].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
              if r.masks is not None else np.zeros((h, w), bool))
        q = ref(ip, verbose=False, device=0, imgsz=512)[0]
        sr = (cv2.resize(q.masks.data[0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
              if q.masks is not None else np.zeros((h, w), bool))
        rows.append([group, centre, modality, stem, dice(sp, gt), int(sp.sum()), dice(sr, gt),
                     dice(sp, sr) if (sp.any() and sr.any()) else 0.0, int(sr.sum())])
with open(a.out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["group", "centre", "modality", "image", "dsc_primary", "px_primary", "dsc_yolo512", "agr_yolo512", "px_yolo512"])
    wr.writerows(rows)
print("Saved", len(rows), "rows to", os.path.abspath(a.out))
