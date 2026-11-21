#!/usr/bin/env python3
"""
Reviewer 6, Comment 5 - referee segmentation quality and error correlation.

For every external image, runs the primary model and each referee exactly as in
the repository inference scripts and records:
  dsc_<model>   Dice of that model's mask against the ground truth
  agr_<ref>     Agreement Dice between the primary mask and the referee mask
                (0 if either mask is empty, as in the paper)
  px_<model>    number of foreground pixels (0 = empty mask)

Models
  primary      YOLOv8n-Seg seed 42         (mask resize: nearest, as 01_external_validation)
  yolo         Independent YOLO seed 123   (mask resize: linear, as 03_generate_external_referee_predictions)
  segformer    SegFormer-B0, 512           (as 01_external_validation)
  unetpp       original UNet++ (trained 256, applied 512, as 23B_generate_unetpp_external_masks_v2)
  unetpp_m     matched UNet++ from R5B_train_unetpp_matched.py (optional; 512)

Usage (Windows, from C:\\seg_uncertain):
  python R5A_referee_quality.py --gpu 1
  (add --unetpp_matched "C:\\seg_uncertain\\unetpp_matched\\checkpoints\\best_unetpp_matched.pth"
   once R5B has finished)
Output: referee_quality.csv
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
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import segmentation_models_pytorch as smp

p = argparse.ArgumentParser()
p.add_argument("--gpu", default="1")
p.add_argument("--primary", default=r"C:\seg_uncertain\phase3_yolo_training\YOLO_POLYP_SEG\weights\best.pt")
p.add_argument("--yolo_ref", default=r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee\YOLO_REFEREE_SEED123\weights\best.pt")
p.add_argument("--segformer", default=r"C:\seg_uncertain\segformer\checkpoints\best_model.pth")
p.add_argument("--unetpp", default=r"C:\Medical_image_analysis\colon_cancer\kvasir-seg\Kvasir-SEG\best_unetplusplus_model.pth")
p.add_argument("--unetpp_matched", default="", help="weights from R5B (optional)")
p.add_argument("--out", default="referee_quality.csv")
a = p.parse_args()

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

if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
DEV = "cuda"
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

primary = YOLO(a.primary)
yolo_ref = YOLO(a.yolo_ref)
proc = SegformerImageProcessor(do_resize=False, do_normalize=True, do_rescale=True)
segf = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True)
segf.load_state_dict(torch.load(a.segformer, map_location=DEV)); segf.to(DEV).eval()


def load_unetpp(path):
    m = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    m.load_state_dict(torch.load(path, map_location=DEV)); return m.to(DEV).eval()


unet = load_unetpp(a.unetpp)
unet_m = load_unetpp(a.unetpp_matched) if a.unetpp_matched else None


def yolo_mask(model, path, h, w, interp):
    r = model(path, verbose=False, device=0)[0]
    if r.masks is None:
        return np.zeros((h, w), bool)
    mk = r.masks.data[0].cpu().numpy()
    return cv2.resize(mk, (w, h), interpolation=interp) > 0.5


def segformer_mask(img):
    h, w = img.shape[:2]
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (512, 512))
    pv = proc(images=x, return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        lg = F.interpolate(segf(pixel_values=pv).logits, size=(h, w), mode="bilinear", align_corners=False)
    return torch.argmax(lg, 1).squeeze().cpu().numpy().astype(bool)


def unet_mask(model, img):
    h, w = img.shape[:2]
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (512, 512)).astype(np.float32) / 255.0
    x = torch.from_numpy((x - MEAN) / STD).permute(2, 0, 1).unsqueeze(0).float().to(DEV)
    with torch.no_grad():
        pr = (torch.sigmoid(model(x)) > 0.5).float().squeeze().cpu().numpy()
    return cv2.resize(pr, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5


def dice(x, y):
    s = x.sum() + y.sum()
    return 2.0 * np.logical_and(x, y).sum() / s if s else 0.0


def agreement(x, y):                      # 0 when either mask is empty (paper convention)
    return dice(x, y) if (x.any() and y.any()) else 0.0


refs = ["yolo", "segformer", "unetpp"] + (["unetpp_m"] if unet_m is not None else [])
header = ["dataset", "image", "dsc_primary", "px_primary"]
for r in refs:
    header += [f"dsc_{r}", f"agr_{r}", f"px_{r}"]

rows = []
for ds, (img_dir, msk_dir) in DATASETS.items():
    for name in tqdm(sorted(os.listdir(img_dir)), desc=ds):
        stem = os.path.splitext(name)[0]
        ip = os.path.join(img_dir, name); gp = os.path.join(msk_dir, stem + ".png")
        if not os.path.exists(gp):
            continue
        img = cv2.imread(ip); h, w = img.shape[:2]
        gt = cv2.resize(cv2.imread(gp, 0), (w, h), interpolation=cv2.INTER_NEAREST) > 127
        sp = yolo_mask(primary, ip, h, w, cv2.INTER_NEAREST)
        masks = {"yolo": yolo_mask(yolo_ref, ip, h, w, cv2.INTER_LINEAR),
                 "segformer": segformer_mask(img),
                 "unetpp": unet_mask(unet, img)}
        if unet_m is not None:
            masks["unetpp_m"] = unet_mask(unet_m, img)
        row = [ds, stem, dice(sp, gt), int(sp.sum())]
        for r in refs:
            row += [dice(masks[r], gt), agreement(sp, masks[r]), int(masks[r].sum())]
        rows.append(row)

with open(a.out, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(header); wr.writerows(rows)
print("Saved", len(rows), "rows to", os.path.abspath(a.out))
