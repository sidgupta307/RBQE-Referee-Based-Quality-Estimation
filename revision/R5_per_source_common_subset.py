#!/usr/bin/env python3
"""
Reviewer 5, Comment 2 - results for each distinct external source on the
common non-empty subset (primary and Independent YOLO masks both non-empty;
975 images in total) and on the non-empty-primary subset, with image-level and
sequence-level tests (methods of R4_sequence_level_analysis.py).
CVC-300 is not a distinct source (all its images are contained in CVC-ColonDB)
and is therefore not reported separately.

Usage:  python R5_per_source_common_subset.py sample_results primary_confidence.csv
"""
import sys
import numpy as np
import R4_sequence_level_analysis as r4

r4.RES, r4.CONF = sys.argv[1], sys.argv[2]
m = r4.assign_clusters(r4.load(), 15, 34)
d = m[m.dataset != "cvc_300"]
SIG = {"SegFormer-B0": lambda t: -t.segformer_score.values, "Indep. YOLO": lambda t: -t.yolo_score.values,
       "Confidence": lambda t: -t.conf_det.values}
for name, cond in [("Common non-empty subset (primary and Independent YOLO masks non-empty)", lambda t: t.p_ne & t.yo_ne),
                   ("Non-empty primary mask", lambda t: t.p_ne)]:
    print(f"\n=== {name}; pooled over all releases: n = {int(cond(m).sum())}, failures = {int(m.failure[cond(m)].sum())}")
    for lab, src in [("CVC-ClinicDB", "cvc_clinicdb"), ("CVC-ColonDB", "cvc_colondb"), ("ETIS-Larib", "etis_larib")]:
        t = d[(d.dataset == src) & cond(d)]; y = t.failure.values; cl = t.cl.values
        print(f"  {lab:12s} n={len(t):4d} failures={y.sum():3d} clusters={len(np.unique(cl))} (with failures {t[t.failure == 1].cl.nunique()}) | "
              + " ".join(f"{k} {r4.auc(y, f(t)):.3f}" for k, f in SIG.items()))
        for a, b in [("SegFormer-B0", "Indep. YOLO"), ("SegFormer-B0", "Confidence")]:
            s1, s2 = SIG[a](t), SIG[b](t)
            _, _, D, _, p, _ = r4.clustered_delong(y, s1, s2, cl); lo, hi = r4.cluster_boot(y, s1, s2, cl)
            print(f"      {a} vs {b:11s} dAUC {D:+.3f}  p_img {r4.image_delong(y, s1, s2):.4f}  p_cluster {p:.4f}  "
                  f"p_jk {r4.cluster_jackknife_p(y, s1, s2, cl):.4f}  clusterCI [{lo:+.3f}, {hi:+.3f}]")
