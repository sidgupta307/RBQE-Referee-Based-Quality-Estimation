#!/usr/bin/env python3
"""
Reviewer 3, Comment 1 - Independent YOLO Referee retrained under the SegFormer-B0
input resolution.

Identical to scripts/01_train_independent_yolo_referee.py (same data split,
pretrained YOLOv8n-Seg initialization, seed 123, AdamW lr 1e-3, weight decay
5e-4, cosine schedule, 50 epochs, patience 10, batch 16, AMP, deterministic)
except imgsz = 512 for training and inference, matching SegFormer-B0
(512 x 512, 50 epochs). The primary model is unchanged (640 x 640, as deployed).

Step 1 trains the referee; step 2 runs it on the 1,248 external images and
records its own DSC and its Agreement Dice with the primary mask, using the same
mask conventions as R5A_referee_quality.py.

Usage (Windows, from C:\\seg_uncertain):
  python R3A_yolo_referee_512.py --gpu 1
  (add --skip_train to rerun only the inference step)
Output: yolo512_referee.csv
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
p.add_argument("--gpu", default="1")
p.add_argument("--data", default=r"C:\seg_uncertain\phase2_yolo_training\dataset.yaml")
p.add_argument("--pretrained", default=r"C:\Polygon_polyp_new(05_26)\yolov8n-seg.pt")
p.add_argument("--project", default=r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee")
p.add_argument("--name", default="YOLO_REFEREE_SEED123_IMG512")
p.add_argument("--primary", default=r"C:\seg_uncertain\phase3_yolo_training\YOLO_POLYP_SEG\weights\best.pt")
p.add_argument("--skip_train", action="store_true")
p.add_argument("--out", default="yolo512_referee.csv")
a = p.parse_args()

if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")

DATASETS = {
    "cvc_clinicdb": (r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images", r"C:\Polygon_polyp_new(05_26)\cvc_dataset\masks"),
    "cvc_colondb": (r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images", r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\masks"),
    "etis_larib": (r"C:\Polygon_polyp_new(05_26)\etis-larib\images", r"C:\Polygon_polyp_new(05_26)\etis-larib\masks"),
    "cvc_300": (r"C:\Polygon_polyp_new(05_26)\CVC-300\images", r"C:\Polygon_polyp_new(05_26)\CVC-300\masks"),
}
weights = os.path.join(a.project, a.name, "weights", "best.pt")


def main():
    # ------------------------------------------------------------ 1. train
    if not a.skip_train:
        model = YOLO(a.pretrained)
        model.train(data=a.data, epochs=50, imgsz=512, batch=16, device=0, workers=8, seed=123,
                    deterministic=True, patience=10, optimizer="AdamW", lr0=1e-3, weight_decay=5e-4,
                    cos_lr=True, amp=True, cache=False, project=a.project, name=a.name, exist_ok=True,
                    save=True, plots=True, verbose=True)
    print("Referee weights:", weights)

    # ------------------------------------------------------------ 2. external inference
    primary = YOLO(a.primary); ref = YOLO(weights)

    def dice(x, y):
        s = x.sum() + y.sum(); return 2.0 * np.logical_and(x, y).sum() / s if s else 0.0

    rows = []
    for ds, (img_dir, msk_dir) in DATASETS.items():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=ds):
            stem = os.path.splitext(name)[0]
            ip = os.path.join(img_dir, name); gp = os.path.join(msk_dir, stem + ".png")
            if not os.path.exists(gp):
                continue
            img = cv2.imread(ip); h, w = img.shape[:2]
            gt = cv2.resize(cv2.imread(gp, 0), (w, h), interpolation=cv2.INTER_NEAREST) > 127
            r = primary(ip, verbose=False, device=0)[0]                     # primary as deployed (640)
            sp = (cv2.resize(r.masks.data[0].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                  if r.masks is not None else np.zeros((h, w), bool))
            q = ref(ip, verbose=False, device=0, imgsz=512)[0]                # referee at 512
            sr = (cv2.resize(q.masks.data[0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                  if q.masks is not None else np.zeros((h, w), bool))
            agr = dice(sp, sr) if (sp.any() and sr.any()) else 0.0
            rows.append([ds, stem, dice(sp, gt), int(sp.sum()), dice(sr, gt), agr, int(sr.sum())])
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "image", "dsc_primary", "px_primary", "dsc_yolo512", "agr_yolo512", "px_yolo512"])
        wr.writerows(rows)
    print("Saved", len(rows), "rows to", os.path.abspath(a.out))


if __name__ == "__main__":      # required on Windows because the trainer starts worker processes
    main()
