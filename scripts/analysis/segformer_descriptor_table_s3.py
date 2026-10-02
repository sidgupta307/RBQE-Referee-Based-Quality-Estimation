"""
Recompute Supplementary Table S3 (SegFormer-B0 descriptor-level failure detection)
from the per-image descriptor file, using the orientation stated in the paper:
  - agreement descriptors (Dice, IoU, Boundary, Area Ratio): lower agreement = higher failure risk
  - Centroid Distance: higher distance = higher failure risk
Failure = ground-truth DSC < 0.50. Threshold metrics at the Youden-optimal point.

Usage (from C:\\seg_uncertain):
    python check_segformer_table_s3.py chat_gpt\\20B_segformer_clean_agreement\\clean_agreement_features.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

path = sys.argv[1]
d = pd.read_csv(path)
gt_col = "gt_dice" if "gt_dice" in d.columns else "dice"
y = (d[gt_col] < 0.50).astype(int).values

print(f"File: {path}")
print(f"Images: {len(d)}   Failures: {y.sum()}   Non-failures: {len(y) - y.sum()}")

deg = (d["centroid_distance"] >= 0.999).values          # one or both masks empty
both_empty = deg & (d["area_ratio"] >= 0.999).values
print(f"Degenerate rows (centroid = 1): {deg.sum()}  "
      f"[failures among them: {y[deg].sum()}, area_ratio==0: {(d.loc[deg,'area_ratio']==0).sum()}, "
      f"both-empty (area_ratio==1): {both_empty.sum()}]")
print(f"Non-degenerate rows: {(~deg).sum()}  (paper reports 1,046)")
print(f"Restricted Agreement Dice ROC-AUC: {roc_auc_score(y[~deg], -d.loc[~deg,'agreement_dice']):.4f}  (paper: 0.876)\n")

rows = []
for col in ["agreement_dice", "agreement_iou", "centroid_distance", "boundary_agreement", "area_ratio"]:
    risk = d[col].values if col == "centroid_distance" else -d[col].values
    auc = roc_auc_score(y, risk)
    fpr, tpr, thr = roc_curve(y, risk)
    j = np.argmax(tpr - fpr)
    pred = (risk >= thr[j]).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rows.append([col, auc, acc, prec, rec, f1, tp, fp, fn, tn])

out = pd.DataFrame(rows, columns=["Descriptor", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1",
                                  "TP", "FP", "FN", "TN"])
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
print("\nPaper Table S3 currently reports: Dice 0.960, IoU 0.960, Centroid 0.951, Boundary 0.949, Area Ratio 0.463")
