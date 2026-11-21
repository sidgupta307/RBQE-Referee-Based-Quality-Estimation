#!/usr/bin/env python3
"""
Reviewer 6, Comment 5 - analysis of referee_quality.csv (output of R5A).

Reports, on the standardized 1,223-image benchmark:
  1. referee segmentation quality (own DSC vs ground truth)
  2. error dependence with the primary model (shared failures, Spearman rho)
  3. agreement ROC-AUC (bootstrap CI) for every referee, full and non-empty
  4. paired comparisons: image-level DeLong and the three sequence-level
     methods of R4 (clustered DeLong, cluster bootstrap, cluster jackknife)
Published agreement scores (merged_referee_scores.csv) are used for the
Independent YOLO and SegFormer-B0 referees; UNet++ values come from R5A.

Usage:  python R5C_analyze_referee_quality.py sample_results primary_confidence.csv referee_quality.csv
Needs R4_sequence_level_analysis.py in the same folder.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import R4_sequence_level_analysis as r4

RES, CONF, QUAL = sys.argv[1], sys.argv[2], sys.argv[3]
r4.RES, r4.CONF = RES, CONF
d = r4.assign_clusters(r4.load(), 15, 34)
q = pd.read_csv(QUAL); q.image = q.image.astype(str)
m = d.merge(q, on=["dataset", "image"], how="left")
assert m.dsc_primary.notna().all() and len(m) == 1223
print("Alignment: max |recomputed primary DSC - published| =", f"{(m.dsc_primary - m.gt_dice).abs().max():.2e}")
m["agr_yolo_pub"] = m.yolo_score; m["agr_segformer_pub"] = m.segformer_score
ne = m[m.p_ne]
REFS = ["yolo", "segformer", "unetpp", "unetpp_m"]


def boot_ci(y, s, B=2000, seed=2026):
    rng = np.random.default_rng(seed); a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a.append(r4.auc(y[i], s[i]))
    return np.percentile(a, [2.5, 97.5])


print("\n1-2. Referee quality and error dependence")
f = ne[ne.failure == 1]
for r in ["primary"] + REFS:
    line = (f"{r:10s} mean DSC full {m['dsc_' + r].mean():.3f} non-empty {ne['dsc_' + r].mean():.3f}")
    if r != "primary":
        shared = int((f["dsc_" + r] < 0.5).sum())
        line += (f" | empty while primary non-empty {int((ne['px_' + r] == 0).sum()):3d}"
                 f" (non-failures {int(((ne['px_' + r] == 0) & (ne.failure == 0)).sum())})"
                 f" | shared failures {shared}/{len(f)} ({shared / len(f):.0%})"
                 f" | Spearman rho (non-empty) {spearmanr(ne.dsc_primary, ne['dsc_' + r])[0]:.2f}")
    print(line)

print("\n3. Agreement ROC-AUC [bootstrap 95% CI]")
cols = {"yolo (published)": "agr_yolo_pub", "segformer (published)": "agr_segformer_pub",
        "yolo (recomputed)": "agr_yolo", "segformer (recomputed)": "agr_segformer",
        "unetpp": "agr_unetpp", "unetpp_m": "agr_unetpp_m"}
for nm, t in [("full", m), ("non-empty", ne)]:
    y = t.failure.values
    for lab, c in cols.items():
        lo, hi = boot_ci(y, -t[c].values)
        print(f"  {nm:9s} {lab:24s} {r4.auc(y, -t[c].values):.3f} [{lo:.3f}, {hi:.3f}]")
t = m[m.p_ne & (m.px_yolo > 0) & (m.px_segformer > 0) & (m.px_unetpp_m > 0)]
print(f"  all masks non-empty (n={len(t)}, failures={int(t.failure.sum())}): "
      + ", ".join(f"{lab} {r4.auc(t.failure.values, -t[c].values):.3f}"
                  for lab, c in [("SegFormer", "agr_segformer_pub"), ("YOLO", "agr_yolo_pub"), ("UNet++ matched", "agr_unetpp_m")]))

print("\n4. Paired comparisons")
pairs = [("unetpp_m", "agr_unetpp_m", "SegFormer", "agr_segformer_pub"),
         ("unetpp_m", "agr_unetpp_m", "YOLO", "agr_yolo_pub"),
         ("unetpp", "agr_unetpp", "YOLO", "agr_yolo_pub"),
         ("unetpp", "agr_unetpp", "SegFormer", "agr_segformer_pub"),
         ("unetpp_m", "agr_unetpp_m", "unetpp", "agr_unetpp")]
for nm, t in [("full", m), ("non-empty", ne)]:
    y = t.failure.values; cl = t.cl.values
    for a, ca, b, cb in pairs:
        s1, s2 = -t[ca].values, -t[cb].values
        A1, A2, D, se, p, K = r4.clustered_delong(y, s1, s2, cl)
        lo, hi = r4.cluster_boot(y, s1, s2, cl)
        print(f"  {nm:9s} {a:8s} vs {b:9s} {A1:.3f}-{A2:.3f}={D:+.3f} p_img={r4.image_delong(y, s1, s2):.4f} "
              f"p_cl={p:.4f} p_jk={r4.cluster_jackknife_p(y, s1, s2, cl):.4f} clusterCI[{lo:+.3f},{hi:+.3f}]")
