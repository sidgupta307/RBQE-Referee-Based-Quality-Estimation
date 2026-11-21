#!/usr/bin/env python3
"""
Reviewer 6, Comment 6 - paired statistical comparison of RBQE with the
no-reference baselines on the same evaluation images.

Signals (higher risk score = more likely failure; fixed before analysis):
  RBQE SegFormer-B0     -Agreement Dice (published scores)
  RBQE Independent YOLO -Agreement Dice (published scores)
  Morphology            -Circularity of the primary mask (as in the paper)
  TTA                   number of the nine TTA conditions with an empty mask
  Primary confidence    -confidence of the top detection (R1 output)

For every RBQE-versus-baseline pair, on the full standardized benchmark
(1,223 images) and on the 1,046 images with a non-empty primary mask:
  * ROC-AUC with image-level bootstrap 95% CI (2,000 resamples, seed 2026)
  * Delta-AUC with paired image-level bootstrap 95% CI
  * paired DeLong p-value (image level), with Holm adjustment over the
    four RBQE-vs-morphology/TTA tests in each subset
  * sequence-level: clustered DeLong p, cluster-bootstrap CI and cluster
    jackknife p (clusters as in R4_sequence_level_analysis.py)

Usage:
  python R6_baseline_paired_tests.py sample_results primary_confidence.csv ^
         morphology_features.csv tta_uncertainty_features.csv
Needs R4_sequence_level_analysis.py in the same folder.
Output: baseline_paired_tests.csv
"""
import sys
import numpy as np
import pandas as pd

import R4_sequence_level_analysis as r4

RES, CONF, MORPH, TTA = sys.argv[1:5]
B, SEED = 2000, 2026
r4.RES, r4.CONF = RES, CONF
d = r4.assign_clusters(r4.load(), 15, 34)

mo = pd.read_csv(MORPH); ta = pd.read_csv(TTA)
for x in (mo, ta):
    x["image"] = x.image.astype(str).str.replace(r"\.[A-Za-z]+$", "", regex=True)
m = (d.merge(mo[["dataset", "image", "circularity", "dice"]], on=["dataset", "image"], how="left")
      .merge(ta[["dataset", "image", "num_zero_predictions"]], on=["dataset", "image"], how="left"))
assert m.circularity.notna().all() and m.num_zero_predictions.notna().all() and len(m) == 1223
print("Alignment: max |morphology-file DSC - published DSC| =", f"{(m.dice - m.gt_dice).abs().max():.1e}")

SIG = {"RBQE SegFormer-B0": lambda t: -t.segformer_score.values,
       "RBQE Independent YOLO": lambda t: -t.yolo_score.values,
       "Morphology (Circularity)": lambda t: -t.circularity.values,
       "TTA (zero predictions)": lambda t: t.num_zero_predictions.values.astype(float),
       "Primary-model confidence": lambda t: -t.conf_det.values}
PAIRS = [("RBQE SegFormer-B0", "Morphology (Circularity)"), ("RBQE SegFormer-B0", "TTA (zero predictions)"),
         ("RBQE Independent YOLO", "Morphology (Circularity)"), ("RBQE Independent YOLO", "TTA (zero predictions)"),
         ("Primary-model confidence", "Morphology (Circularity)"), ("Primary-model confidence", "TTA (zero predictions)")]


def boot(y, s1, s2=None):
    rng = np.random.default_rng(SEED); a, dd = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a1 = r4.auc(y[i], s1[i]); a.append(a1)
            if s2 is not None:
                dd.append(a1 - r4.auc(y[i], s2[i]))
    return np.percentile(a, [2.5, 97.5]), (np.percentile(dd, [2.5, 97.5]) if dd else None)


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


rows = []
for name, t in [("Full benchmark", m), ("Primary non-empty", m[m.p_ne])]:
    y = t.failure.values.astype(int); cl = t.cl.values
    print(f"\n=== {name}: n = {len(t)}, failures = {y.sum()}, clusters = {len(np.unique(cl))}")
    for s, f in SIG.items():
        (lo, hi), _ = boot(y, f(t))
        print(f"  {s:26s} ROC-AUC {r4.auc(y, f(t)):.3f} [{lo:.3f}, {hi:.3f}]")
    block = []
    for a, b in PAIRS:
        s1, s2 = SIG[a](t), SIG[b](t)
        A1, A2, D, se, p_cl, K = r4.clustered_delong(y, s1, s2, cl)
        _, (dlo, dhi) = boot(y, s1, s2)
        clo, chi = r4.cluster_boot(y, s1, s2, cl)
        block.append(dict(subset=name, n=len(t), failures=int(y.sum()), comparison=f"{a} vs {b}",
                          auc_a=A1, auc_b=A2, diff=D, boot_lo=dlo, boot_hi=dhi,
                          p_image=r4.image_delong(y, s1, s2), cluster_boot_lo=clo, cluster_boot_hi=chi,
                          p_cluster=p_cl, p_jackknife=r4.cluster_jackknife_p(y, s1, s2, cl)))
    adj = holm(np.array([r["p_image"] for r in block[:4]]))
    for k, r in enumerate(block):
        r["p_image_holm"] = adj[k] if k < 4 else np.nan
        print(f"  {r['comparison']:52s} dAUC {r['diff']:+.3f} [{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}] "
              f"p_img {r['p_image']:.2e} (Holm {r['p_image_holm']:.2e}) | seq: CI [{r['cluster_boot_lo']:+.3f}, "
              f"{r['cluster_boot_hi']:+.3f}] p_cl {r['p_cluster']:.4f} p_jk {r['p_jackknife']:.4f}")
    rows += block

pd.DataFrame(rows).to_csv("baseline_paired_tests.csv", index=False)
print("\nSaved baseline_paired_tests.csv")
