#!/usr/bin/env python3
"""
Reviewer 6, Comment 7 - lesion and imaging-condition descriptors for the four
external benchmark releases (CPU only, no models).

Per image (computed on the original image grid; ground truth used only to
measure lesion size for the stratified evaluation):
  width, height       image resolution
  gt_frac             polyp area as a fraction of the image (lesion size)
  brightness          mean V channel (HSV), 0-255
  contrast            standard deviation of the grey image, 0-255
  sharpness           variance of the Laplacian of the grey image resized to
                      512 px width (higher = sharper; low = blur)
  specular_frac       fraction of pixels with V > 230 and S < 40 (specular
                      highlights)

Usage (Windows, from C:\\seg_uncertain):  python R7A_image_conditions.py
Output: image_conditions.csv
"""
import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

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


def describe(img, gt):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (512, int(round(512 * h / w))), interpolation=cv2.INTER_AREA)
    return dict(width=w, height=h, gt_frac=float(gt.mean()),
                brightness=float(hsv[..., 2].mean()), contrast=float(grey.std()),
                sharpness=float(cv2.Laplacian(small, cv2.CV_64F).var()),
                specular_frac=float(((hsv[..., 2] > 230) & (hsv[..., 1] < 40)).mean()))


if __name__ == "__main__":
    rows = []
    for ds, (img_dir, msk_dir) in DATASETS.items():
        for name in tqdm(sorted(os.listdir(img_dir)), desc=ds):
            stem = os.path.splitext(name)[0]
            gp = os.path.join(msk_dir, stem + ".png")
            if not os.path.exists(gp):
                continue
            img = cv2.imread(os.path.join(img_dir, name)); h, w = img.shape[:2]
            gt = cv2.resize(cv2.imread(gp, 0), (w, h), interpolation=cv2.INTER_NEAREST) > 127
            rows.append(dict(dataset=ds, image=stem, **describe(img, gt)))
    with open("image_conditions.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("Saved", len(rows), "rows to", os.path.abspath("image_conditions.csv"))
