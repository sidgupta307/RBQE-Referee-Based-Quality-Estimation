#!/usr/bin/env python3
"""
Values added to Supplementary Tables S12 and S20 after internal review:
  * the 512 x 512 Independent YOLO Referee row of Table S12 (mean DSC, empty
    referee masks, shared failures, Spearman rho) and the PolypDB shared-failure
    column for every referee;
  * failures accepted at equal coverage (60% of images with the highest
    agreement) for every referee, on the benchmark and on PolypDB (Table S20).

Usage:
  python R12_referee_extras.py sample_results primary_confidence.csv referee_quality.csv ^
         yolo512_referee.csv polypdb_results.csv polypdb_yolo512.csv
Needs R4_sequence_level_analysis.py in the same folder.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import R4_sequence_level_analysis as r4

r4.RES, r4.CONF, RQ, Y5, PDB, P5 = sys.argv[1:7]
d = r4.load()
q = pd.read_csv(RQ); q["image"] = q.image.astype(str)
y5 = pd.read_csv(Y5); y5["image"] = y5.image.astype(str)
m = d.merge(q, on=["dataset", "image"]).merge(y5[["dataset", "image", "dsc_yolo512", "agr_yolo512", "px_yolo512"]], on=["dataset", "image"])
ne = m[m.p_ne]; f = ne[ne.failure == 1]
print("Table S12, benchmark (1,223 images; non-empty primary 1,046; 59 failures)")
for c, lab in [("yolo", "Independent YOLO 640"), ("yolo512", "Independent YOLO 512"), ("segformer", "SegFormer-B0"),
               ("unetpp", "UNet++ original"), ("unetpp_m", "UNet++ matched")]:
    print(f"  {lab:22s} mean DSC full {m['dsc_' + c].mean():.3f} non-empty {ne['dsc_' + c].mean():.3f}  "
          f"empty referee masks {(ne['px_' + c] == 0).sum():3d}  shared failures {(f['dsc_' + c] < 0.5).sum()}/59  "
          f"Spearman rho {spearmanr(ne['dsc_' + c], ne.gt_dice).correlation:.2f}")

p = pd.read_csv(PDB); p = p[p.group == "main"]; p["image"] = p.image.astype(str)
p5 = pd.read_csv(P5); p5 = p5[p5.group == "main"]; p5["image"] = p5.image.astype(str)
p = p.merge(p5[["centre", "modality", "image", "dsc_yolo512", "agr_yolo512"]], on=["centre", "modality", "image"])
pf = p[(p.px_primary > 0) & (p.dsc_primary < 0.5)]
print("\nTable S12, PolypDB shared failures (49 non-empty primary failures)")
for c in ["yolo", "yolo512", "segformer", "unetpp", "unetpp_m"]:
    print(f"  {c:10s} {(pf['dsc_' + c] < 0.5).sum()}/{len(pf)}")


def accepted_failures(score, y, frac=0.6, reps=200, seed=0):
    rng = np.random.default_rng(seed); k = int(round(frac * len(y)))
    return np.mean([y[np.lexsort((rng.random(len(y)), -score))[:k]].sum() for _ in range(reps)]), k


print("\nTable S20, failures among the 60% of images with the highest agreement")
for name, t, y, cols in [("benchmark", m, m.failure.values,
                          {"SegFormer-B0": "segformer_score", "UNet++ matched": "agr_unetpp_m", "UNet++ original": "agr_unetpp",
                           "YOLO 512": "agr_yolo512", "YOLO 640": "yolo_score"}),
                         ("PolypDB", p, (p.dsc_primary < 0.5).astype(int).values,
                          {"SegFormer-B0": "agr_segformer", "UNet++ matched": "agr_unetpp_m", "UNet++ original": "agr_unetpp",
                           "YOLO 512": "agr_yolo512", "YOLO 640": "agr_yolo"})]:
    for lab, c in cols.items():
        v, k = accepted_failures(t[c].values, y)
        print(f"  {name:9s} {lab:16s} accepted {k} images, failures {v:.1f}")
