#!/usr/bin/env python3
"""
Check which Kvasir-SEG split each trained model actually used.

Prints, without changing anything:
  1. image/label counts in phase2_yolo_training (train / val / test) and the
     split counts in split_manifest.csv, plus file-name overlap between splits
  2. for every YOLO run: the dataset, image size, seed and date stored inside
     best.pt and args.yaml, and the number of epochs in results.csv
  3. the dates of best.pt versus the dates of the split folders
  4. primary-model Dice on the current train / val / test images
     (if val/test Dice is as high as train Dice, the primary model may have
     seen those images during training, i.e. it was trained on another split)
  5. optionally, the same counts for the folder of the previous paper
     (--old_root) and how many of its train images are in the current val/test

Usage (Windows, from C:\\seg_uncertain):
  python R0_check_kvasir_split.py --gpu 1
  python R0_check_kvasir_split.py --gpu 1 --old_root "D:\\previous_paper\\yolo_dataset"
"""
import argparse
import datetime as dt
import glob
import os
import sys

_gpu = "1"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--gpu", default="1")
p.add_argument("--root", default=r"C:\seg_uncertain\phase2_yolo_training")
p.add_argument("--old_root", default="", help="optional: dataset folder of the previous paper")
p.add_argument("--runs", nargs="*", default=[
    r"C:\seg_uncertain\phase3_yolo_training\YOLO_POLYP_SEG",
    r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee\YOLO_REFEREE_SEED123",
    r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee\YOLO_REFEREE_SEED123_IMG512"])
p.add_argument("--skip_dice", action="store_true")
a = p.parse_args()

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
stamp = lambda f: dt.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")


def split_files(root):
    out = {}
    for s in ("train", "val", "test"):
        d = os.path.join(root, "images", s)
        if os.path.isdir(d):
            out[s] = sorted(f for f in os.listdir(d) if f.lower().endswith(EXT))
    return out


def report_root(root, title):
    print(f"\n=== {title}: {root}")
    if not os.path.isdir(root):
        print("  folder not found"); return {}
    sp = split_files(root)
    for s, files in sp.items():
        ld = os.path.join(root, "labels", s)
        nl = len(glob.glob(os.path.join(ld, "*.txt"))) if os.path.isdir(ld) else 0
        print(f"  {s:5s}: {len(files):4d} images, {nl:4d} labels   (folder date {stamp(os.path.join(root, 'images', s))})")
    stems = {s: {os.path.splitext(f)[0] for f in v} for s, v in sp.items()}
    for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
        if x in stems and y in stems:
            print(f"  overlap {x}/{y}: {len(stems[x] & stems[y])}")
    for f in glob.glob(os.path.join(root, "*.yaml")) + glob.glob(os.path.join(root, "*.csv")):
        print(f"  file {os.path.basename(f)} (date {stamp(f)})")
        if f.endswith(".yaml"):
            print("    " + open(f).read().replace("\n", "\n    ").strip())
        elif "split" in os.path.basename(f).lower():
            import pandas as pd
            m = pd.read_csv(f)
            col = [c for c in m.columns if c.lower() == "split"]
            if col:
                print("    split counts:", m[col[0]].value_counts().to_dict())
    return stems


cur = report_root(a.root, "Current development folder (used in this paper)")

print("\n=== Trained YOLO runs")
for r in a.runs:
    w = os.path.join(r, "weights", "best.pt")
    print(f"\n  {r}")
    if not os.path.exists(w):
        print("    best.pt not found"); continue
    print(f"    best.pt date: {stamp(w)}")
    try:
        import torch
        ck = torch.load(w, map_location="cpu", weights_only=False)
        ta = ck.get("train_args", {}) or {}
        print(f"    stored in best.pt: date={ck.get('date')}  data={ta.get('data')}  imgsz={ta.get('imgsz')}  "
              f"seed={ta.get('seed')}  epochs={ta.get('epochs')}  best epoch={ck.get('epoch')}")
    except Exception as e:
        print("    could not read best.pt:", e)
    ay = os.path.join(r, "args.yaml")
    if os.path.exists(ay):
        keep = [l.strip() for l in open(ay) if l.split(":")[0] in ("data", "imgsz", "seed", "epochs", "project", "name")]
        print("    args.yaml:", "; ".join(keep))
    rc = os.path.join(r, "results.csv")
    if os.path.exists(rc):
        print(f"    results.csv: {sum(1 for _ in open(rc)) - 1} epochs logged")

if a.old_root:
    old = report_root(a.old_root, "Previous paper folder")
    if old and cur:
        for s in ("val", "test"):
            if "train" in old and s in cur:
                print(f"  previous-paper TRAIN images that are in the current {s.upper()} split: "
                      f"{len(old['train'] & cur[s])} of {len(cur[s])}")

if not a.skip_dice and cur:
    import cv2
    import torch
    from ultralytics import YOLO
    w = os.path.join(a.runs[0], "weights", "best.pt")
    model = YOLO(w)
    print(f"\n=== Primary-model Dice on the current splits ({w})")

    def poly_mask(lbl, h, w_):
        m = np.zeros((h, w_), np.uint8)
        if os.path.exists(lbl):
            for line in open(lbl):
                v = line.split()
                if len(v) > 6:
                    pts = (np.array(v[1:], float).reshape(-1, 2) * [w_, h]).astype(np.int32)
                    cv2.fillPoly(m, [pts], 1)
        return m.astype(bool)

    for s in ("train", "val", "test"):
        files = sorted(glob.glob(os.path.join(a.root, "images", s, "*")))
        files = [f for f in files if f.lower().endswith(EXT)]
        if not files:
            continue
        ds = []
        for f in files:
            img = cv2.imread(f); h, w_ = img.shape[:2]
            gt = poly_mask(os.path.join(a.root, "labels", s, os.path.splitext(os.path.basename(f))[0] + ".txt"), h, w_)
            r = model(f, verbose=False, device=0)[0]
            pr = (cv2.resize(r.masks.data[0].cpu().numpy(), (w_, h), interpolation=cv2.INTER_NEAREST) > 0.5
                  if r.masks is not None else np.zeros((h, w_), bool))
            sm = pr.sum() + gt.sum()
            ds.append(2 * np.logical_and(pr, gt).sum() / sm if sm else 1.0)
        ds = np.array(ds)
        print(f"  {s:5s}: n={len(ds):4d}  mean Dice {ds.mean():.3f}  median {np.median(ds):.3f}  Dice<0.5: {(ds < 0.5).sum()}")
    print("  If val and test Dice are clearly lower than train Dice, the primary model did not train on them.")
