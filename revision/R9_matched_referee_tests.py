#!/usr/bin/env python3
"""
Paired comparisons on the 1,046 images with a non-empty primary mask for the
referees added during revision (protocol-matched UNet++ and the 512 x 512
Independent YOLO Referee), as reported in Supplementary Table S11.

Usage (run from the folder that holds these files; arguments optional):
  python R9_matched_referee_tests.py sample_results primary_confidence.csv referee_quality.csv yolo512_referee.csv
Needs R4_sequence_level_analysis.py in the same folder.
"""
import sys
import numpy as np
import pandas as pd
import R4_sequence_level_analysis as r4

args = sys.argv[1:5] if len(sys.argv) >= 5 else ["sample_results", "primary_confidence.csv",
                                                  "referee_quality.csv", "yolo512_referee.csv"]
r4.RES, r4.CONF, RQ, Y5 = args
d0 = r4.load()
q = pd.read_csv(RQ); q["image"] = q.image.astype(str)
y5 = pd.read_csv(Y5); y5["image"] = y5.image.astype(str)
d0 = (d0.merge(q[["dataset", "image", "agr_unetpp_m"]], on=["dataset", "image"])
        .merge(y5[["dataset", "image", "agr_yolo512"]], on=["dataset", "image"]))
assert len(d0) == 1223
PAIRS = [("UNet++ matched", "SegFormer"), ("UNet++ matched", "YOLO"), ("UNet++ matched", "Confidence"),
         ("SegFormer", "YOLO 512"), ("YOLO 512", "YOLO")]
for kc, ke, tag in [(15, 34, "main"), (8, 17, "coarse")]:
    t = r4.assign_clusters(d0, kc, ke); t = t[t.p_ne]
    y = t.failure.values.astype(int); cl = t.cl.values
    S = {"SegFormer": -t.segformer_score.values, "UNet++ matched": -t.agr_unetpp_m.values,
         "YOLO": -t.yolo_score.values, "YOLO 512": -t.agr_yolo512.values, "Confidence": -t.conf_det.values}
    print(f"=== {tag} clustering: n = {len(t)}, failures = {y.sum()}, clusters = {t.cl.nunique()}")
    for a, b in PAIRS:
        _, _, D, _, pc, _ = r4.clustered_delong(y, S[a], S[b], cl)
        line = f"  {a:15s} vs {b:10s} dAUC {D:+.3f}  p_img {r4.image_delong(y, S[a], S[b]):.4f}  p_cluster {pc:.4f}"
        if tag == "main":
            lo, hi = r4.cluster_boot(y, S[a], S[b], cl)
            line += f"  p_jk {r4.cluster_jackknife_p(y, S[a], S[b], cl):.4f}  clusterCI [{lo:+.3f}, {hi:+.3f}]"
        print(line)
