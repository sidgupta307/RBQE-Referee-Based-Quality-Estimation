"""
Reproduces the referee-comparison statistics reported in the manuscript from the
released per-image files:

  * Full benchmark (1,223 images): SegFormer-B0 vs Independent YOLO Referee,
    paired DeLong test on Agreement Dice  -> dAUC = 0.0370, p = 6.32e-5
  * Independent YOLO restricted subset (both masks non-empty, 975 images) -> 0.783
  * Secondary paired analysis on the identical 975 images
    -> SegFormer-B0 0.885 vs Independent YOLO 0.783, paired DeLong p = 0.0089

Inputs (sample_results/):
  merged_referee_scores.csv   columns: dataset, image, gt_dice, segformer_score, yolo_score, failure
  merged_1223_yolo_seed.csv   per-image Independent YOLO descriptors (incl. centroid_distance)

Failure = ground-truth DSC < 0.50. Lower agreement = higher failure risk.

Usage:
    python scripts/analysis/paired_delong_and_restricted.py sample_results
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def delong_paired(y, s1, s2):
    """Paired DeLong test for two correlated ROC AUCs (DeLong et al., 1988)."""
    y = np.asarray(y).astype(bool)
    def components(s):
        pos, neg = s[y], s[~y]
        v10 = np.array([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
        v01 = np.array([(pos < n).mean() + 0.5 * (pos == n).mean() for n in neg])
        return v10.mean(), v10, v01
    a1, v10a, v01a = components(np.asarray(s1))
    a2, v10b, v01b = components(np.asarray(s2))
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    cov = s10 / y.sum() + s01 / (~y).sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, 2 * norm.sf(abs(z))


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample_results"
    scores = pd.read_csv(os.path.join(folder, "merged_referee_scores.csv"))
    yolo = pd.read_csv(os.path.join(folder, "merged_1223_yolo_seed.csv"))
    m = scores.merge(yolo[["dataset", "image", "centroid_distance"]], on=["dataset", "image"])

    y = m["failure"].values
    seg = -m["segformer_score"].values      # lower agreement = higher risk
    yol = -m["yolo_score"].values

    print(f"Images: {len(m)}  Failures: {y.sum()}  Non-failures: {len(y) - y.sum()}")

    a1, a2, p = delong_paired(y, seg, yol)
    print(f"\nFull benchmark: SegFormer-B0 {a1:.4f} vs Independent YOLO {a2:.4f}  "
          f"dAUC = {a1 - a2:.4f}  paired DeLong p = {p:.3g}")

    both_nonempty = m["centroid_distance"].values < 0.999999   # Independent YOLO both-non-empty stratum
    yr = y[both_nonempty]
    print(f"\nIndependent YOLO restricted subset: n = {both_nonempty.sum()}, failures = {yr.sum()}, "
          f"ROC-AUC = {roc_auc_score(yr, yol[both_nonempty]):.4f}")

    a1, a2, p = delong_paired(yr, seg[both_nonempty], yol[both_nonempty])
    print(f"Paired analysis on the same {both_nonempty.sum()} images: SegFormer-B0 {a1:.5f} vs "
          f"Independent YOLO {a2:.5f}  dAUC = {a1 - a2:.5f}  paired DeLong p = {p:.5f}")
