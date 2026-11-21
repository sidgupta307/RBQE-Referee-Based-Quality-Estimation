#!/usr/bin/env python3
"""
Reviewer 6, Comment 7 - additional external validation on PolypDB
(Jha et al., 2024; https://osf.io/pr7ms/).

Folder layout used: <root>/PolypDB_center_wise/<centre>/<modality>/{images,masks}.
Main external set: BKAI (Vietnam: WLI, BLI, FICE, LCI) and Karolinska (Sweden:
WLI). The Norwegian Simula centre shares its country of origin with Kvasir-SEG
(the training set of every model): its WLI images are excluded, and its NBI
images (a modality absent from Kvasir-SEG) are processed as a separate,
clearly labelled sensitivity group. No model is retrained or tuned.

Steps
 1. Overlap check: pixel-level MD5 (decoded image array) of every PolypDB image
    against Kvasir-SEG and the four external benchmark releases, plus a
    perceptual difference hash (dHash, Hamming distance <= 4) to flag possible
    near-duplicates. Exact duplicates are excluded; near-duplicate flags are
    reported in the CSV.
 2. Imaging modality and centre are taken from the folder path.
 3. Inference exactly as in R5A_referee_quality.py: primary YOLOv8n-Seg (with
    top-detection confidence), Independent YOLO Referee, SegFormer-B0, original
    UNet++ and (optional) protocol-matched UNet++. Records DSC vs ground truth,
    Agreement Dice, mask sizes, Circularity of the primary mask (morphology
    baseline, same definition as the paper) and image-condition descriptors.

Usage (Windows, from C:\\seg_uncertain):
  python R7B_polypdb_external.py --polypdb "C:\\PolypDB\\PolypDB\\PolypDB" --gpu 1 ^
      --unetpp_matched "C:\\seg_uncertain\\unetpp_matched\\checkpoints\\best_unetpp_matched.pth"
Output: polypdb_results.csv and polypdb_overlap_report.csv
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
import hashlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import segmentation_models_pytorch as smp

p = argparse.ArgumentParser()
p.add_argument("--polypdb", required=True, help="folder that contains PolypDB_center_wise")
p.add_argument("--main", default="BKAI/WLI,BKAI/BLI,BKAI/FICE,BKAI/LCI,Karolinska/WLI",
               help="centre/modality folders of the main external set")
p.add_argument("--sensitivity", default="Simula/NBI", help="centre/modality folders of the sensitivity group")
p.add_argument("--kvasir_images", default=r"C:\Polygon_polyp_new(05_26)\kvasir_dataset\images")
p.add_argument("--external_roots", default=r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images;"
               r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images;C:\Polygon_polyp_new(05_26)\etis-larib\images;"
               r"C:\Polygon_polyp_new(05_26)\CVC-300\images")
p.add_argument("--gpu", default="1")
p.add_argument("--primary", default=r"C:\seg_uncertain\phase3_yolo_training\YOLO_POLYP_SEG\weights\best.pt")
p.add_argument("--yolo_ref", default=r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee\YOLO_REFEREE_SEED123\weights\best.pt")
p.add_argument("--segformer", default=r"C:\seg_uncertain\segformer\checkpoints\best_model.pth")
p.add_argument("--unetpp", default=r"C:\Medical_image_analysis\colon_cancer\kvasir-seg\Kvasir-SEG\best_unetplusplus_model.pth")
p.add_argument("--unetpp_matched", default="")
p.add_argument("--out", default="polypdb_results.csv")
p.add_argument("--check", action="store_true", help="only list the folders that will be used, then exit")
a = p.parse_args()

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)) if os.path.isdir(folder) else []


def find_sub(root, names):
    for n in names:
        q = os.path.join(root, n)
        if os.path.isdir(q):
            return q
    return None


def pix_md5(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_mask(path, h, w):
    m = cv2.imread(path, cv2.IMREAD_COLOR)            # colour-safe: any channel > 127 is foreground
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.max(axis=2) > 127


# ---------------------------------------------------------------- 0. folder check
p_check = "--check" in sys.argv
cw0 = os.path.join(a.polypdb, "PolypDB_center_wise")
print("Folder check:", cw0)
for grp, lst in (("main", a.main), ("sensitivity", a.sensitivity)):
    for cm0 in [x for x in lst.split(",") if x]:
        r0 = os.path.join(cw0, *cm0.split("/"))
        i0 = find_sub(r0, ["images", "Images"]); m0 = find_sub(r0, ["masks", "annotations", "Masks", "Annotations"])
        print(f"  {grp:11s} {cm0:16s} images: {len(list_images(i0)) if i0 else 'NOT FOUND':>5}   masks: {len(list_images(m0)) if m0 else 'NOT FOUND':>5}")
if p_check:
    sys.exit("Check only (--check): no models were run.")

# ---------------------------------------------------------------- 1. reference hashes
print("Hashing reference sets (Kvasir-SEG and the four external releases)...")
ref_md5, ref_dh = {}, []
for root in [a.kvasir_images] + a.external_roots.split(";"):
    tag = "kvasir" if root == a.kvasir_images else os.path.basename(os.path.dirname(root))
    for f in tqdm(list_images(root), desc=tag):
        im = cv2.imread(os.path.join(root, f))
        if im is None:
            continue
        ref_md5[pix_md5(im)] = f"{tag}/{f}"; ref_dh.append((dhash(im), f"{tag}/{f}"))
ref_dh_arr = np.array([d for d, _ in ref_dh], dtype=np.uint64)


# ---------------------------------------------------------------- 3. models
if not torch.cuda.is_available():
    sys.exit(f"GPU {a.gpu} is not available to PyTorch; check the index with nvidia-smi.")
DEV = "cuda"
print(f"Using nvidia-smi GPU {a.gpu}: {torch.cuda.get_device_name(0)}")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
primary = YOLO(a.primary); yolo_ref = YOLO(a.yolo_ref)
proc = SegformerImageProcessor(do_resize=False, do_normalize=True, do_rescale=True)
segf = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True)
segf.load_state_dict(torch.load(a.segformer, map_location=DEV)); segf.to(DEV).eval()


def load_unetpp(path):
    m = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    m.load_state_dict(torch.load(path, map_location=DEV)); return m.to(DEV).eval()


unet = load_unetpp(a.unetpp); unet_m = load_unetpp(a.unetpp_matched) if a.unetpp_matched else None


def run_primary(path, h, w):
    r = primary(path, verbose=False, device=0)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    conf = float(r.boxes.conf[0]) if n else 0.0
    if r.masks is None:
        return np.zeros((h, w), bool), n, conf
    return cv2.resize(r.masks.data[0].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5, n, conf


def run_yolo_ref(path, h, w):
    r = yolo_ref(path, verbose=False, device=0)[0]
    if r.masks is None:
        return np.zeros((h, w), bool)
    return cv2.resize(r.masks.data[0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def run_segformer(img):
    h, w = img.shape[:2]
    pv = proc(images=cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (512, 512)), return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        lg = F.interpolate(segf(pixel_values=pv).logits, size=(h, w), mode="bilinear", align_corners=False)
    return torch.argmax(lg, 1).squeeze().cpu().numpy().astype(bool)


def run_unet(model, img):
    h, w = img.shape[:2]
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (512, 512)).astype(np.float32) / 255.0
    x = torch.from_numpy((x - MEAN) / STD).permute(2, 0, 1).unsqueeze(0).float().to(DEV)
    with torch.no_grad():
        pr = (torch.sigmoid(model(x)) > 0.5).float().squeeze().cpu().numpy()
    return cv2.resize(pr, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5


def dice(x, y):
    s = x.sum() + y.sum(); return 2.0 * np.logical_and(x, y).sum() / s if s else 0.0


def agreement(x, y):
    return dice(x, y) if (x.any() and y.any()) else 0.0


def circularity(mask):                     # same definition as scripts/baselines/03_extract_morphology_features.py
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    per = cv2.arcLength(max(cs, key=cv2.contourArea), True)
    return float(4 * np.pi * m.sum() / (per ** 2 + 1e-8)) if per > 0 else 0.0


def conditions(img, gt):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (512, int(round(512 * h / w))), interpolation=cv2.INTER_AREA)
    return [w, h, float(gt.mean()), float(hsv[..., 2].mean()), float(grey.std()),
            float(cv2.Laplacian(small, cv2.CV_64F).var()), float(((hsv[..., 2] > 230) & (hsv[..., 1] < 40)).mean())]


# ---------------------------------------------------------------- 4. run
refs = ["yolo", "segformer", "unetpp"] + (["unetpp_m"] if unet_m is not None else [])
header = ["group", "centre", "modality", "image", "width", "height", "gt_frac", "brightness", "contrast", "sharpness",
          "specular_frac", "near_duplicate_of", "dsc_primary", "n_det", "conf_det", "px_primary", "circularity"]
for r in refs:
    header += [f"dsc_{r}", f"agr_{r}", f"px_{r}"]
rows, overlap = [], []
cw = os.path.join(a.polypdb, "PolypDB_center_wise")
if not os.path.isdir(cw):
    sys.exit(f"Folder not found: {cw}")
jobs = [("main", x) for x in a.main.split(",") if x] + [("sensitivity", x) for x in a.sensitivity.split(",") if x]
for group, cm in jobs:
    centre, modality = cm.split("/")
    croot = os.path.join(cw, centre, modality)
    idir = find_sub(croot, ["images", "Images"]); mdir = find_sub(croot, ["masks", "annotations", "Masks", "Annotations"])
    if idir is None or mdir is None:
        print(f"WARNING: images/masks folder not found under {croot}"); continue
    mfiles = {os.path.splitext(f)[0]: f for f in list_images(mdir)}
    for f in tqdm(list_images(idir), desc=cm):
        stem = os.path.splitext(f)[0]
        if stem not in mfiles:
            overlap.append([cm, f, "no matching mask", ""]); continue
        ip = os.path.join(idir, f); img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]; md = pix_md5(img)
        if md in ref_md5:
            overlap.append([cm, f, "exact duplicate (excluded)", ref_md5[md]]); continue
        hd = dhash(img)
        ham = np.array([bin(int(hd) ^ int(x)).count("1") for x in ref_dh_arr]) if len(ref_dh_arr) else np.array([])
        near = ref_dh[int(ham.argmin())][1] if len(ham) and ham.min() <= 4 else ""
        if near:
            overlap.append([cm, f, "possible near-duplicate (kept, flagged)", near])
        gt = read_mask(os.path.join(mdir, mfiles[stem]), h, w)
        if not gt.any():
            overlap.append([cm, f, "empty ground truth (excluded)", ""]); continue
        sp, n, conf = run_primary(ip, h, w)
        masks = {"yolo": run_yolo_ref(ip, h, w), "segformer": run_segformer(img), "unetpp": run_unet(unet, img)}
        if unet_m is not None:
            masks["unetpp_m"] = run_unet(unet_m, img)
        row = [group, centre, modality, stem] + conditions(img, gt) + [near, dice(sp, gt), n, conf, int(sp.sum()), circularity(sp)]
        for r in refs:
            row += [dice(masks[r], gt), agreement(sp, masks[r]), int(masks[r].sum())]
        rows.append(row)

with open(a.out, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(header); wr.writerows(rows)
with open("polypdb_overlap_report.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["centre_modality", "image", "status", "matched_reference"]); wr.writerows(overlap)
print(f"\nSaved {len(rows)} images to {os.path.abspath(a.out)}")
print("Overlap report:", {s: sum(1 for o in overlap if o[2] == s) for s in set(o[2] for o in overlap)})
print("Images per group/centre/modality:")
from collections import Counter
for k, v in sorted(Counter((r[0], r[1], r[2]) for r in rows).items()):
    print("  ", k, v)
