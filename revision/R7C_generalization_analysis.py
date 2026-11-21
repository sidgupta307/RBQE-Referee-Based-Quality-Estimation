#!/usr/bin/env python3
"""
Reviewer 6, Comment 7 - generalization analysis. Reproduces every number in the
response to this comment:
  A. per-source results on the standardized benchmark (all signals)
  B. external validation on PolypDB (output of R7B_polypdb_external.py):
     main set (BKAI + Karolinska) and the Simula-NBI sensitivity group, with
     image-level tests and case-level clustering (BKAI case code in the file
     name, e.g. "..._BN005_002" -> case "..._BN005"; other images are their
     own cluster), results by imaging modality, and a sensitivity analysis
     excluding images flagged as possible near-duplicates
  C. failure detection across lesion-size and imaging-condition tertiles
     (output of R7A_image_conditions.py)

Usage:
  python R7C_generalization_analysis.py sample_results primary_confidence.csv ^
         morphology_features.csv tta_uncertainty_features.csv ^
         polypdb_results.csv image_conditions.csv
Needs R4_sequence_level_analysis.py in the same folder.
"""
import sys
import numpy as np
import pandas as pd

import R4_sequence_level_analysis as r4

RES, CONF, MORPH, TTA, PDB, COND = sys.argv[1:7]
r4.RES, r4.CONF = RES, CONF
B, SEED = 2000, 2026


def boot_ci(y, s):
    rng = np.random.default_rng(SEED); a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a.append(r4.auc(y[i], s[i]))
    return np.percentile(a, [2.5, 97.5])


def tests(t, pairs, sig):
    y = t.failure.values.astype(int); cl = t.cl.values
    for a, b in pairs:
        s1, s2 = sig[a](t), sig[b](t)
        _, _, D, _, p_cl, _ = r4.clustered_delong(y, s1, s2, cl)
        lo, hi = r4.cluster_boot(y, s1, s2, cl)
        print(f"      {a:10s} vs {b:10s} dAUC {D:+.3f}  p_img {r4.image_delong(y, s1, s2):.4f}  "
              f"p_cluster {p_cl:.4f}  p_jk {r4.cluster_jackknife_p(y, s1, s2, cl):.4f}  clusterCI [{lo:+.3f}, {hi:+.3f}]")


# ------------------------------------------------------------------ A. per source
d = r4.assign_clusters(r4.load(), 15, 34)
mo = pd.read_csv(MORPH); ta = pd.read_csv(TTA)
for x in (mo, ta):
    x["image"] = x.image.astype(str).str.replace(r"\.[A-Za-z]+$", "", regex=True)
m = (d.merge(mo[["dataset", "image", "circularity"]], on=["dataset", "image"])
      .merge(ta[["dataset", "image", "num_zero_predictions"]], on=["dataset", "image"]))
SIG = {"SegFormer": lambda t: -t.segformer_score.values, "YOLO": lambda t: -t.yolo_score.values,
       "Confidence": lambda t: -t.conf_det.values, "Morphology": lambda t: -t.circularity.values,
       "TTA": lambda t: t.num_zero_predictions.values.astype(float)}
print("A. Per-source results on the standardized benchmark (ROC-AUC)")
for name, ds in [("CVC-ClinicDB", "cvc_clinicdb"), ("CVC-ColonDB", "cvc_colondb"),
                 ("ETIS-Larib", "etis_larib"), ("CVC-300", "cvc_300")]:
    for sub in ("full", "non-empty"):
        t = m[(m.dataset == ds) & (m.p_ne if sub == "non-empty" else True)]
        y = t.failure.values
        line = f"  {name:12s} {sub:9s} n={len(t):4d} failures={y.sum():3d} mean DSC={t.gt_dice.mean():.3f} |"
        if 1 < y.sum() < len(y) - 1:
            line += " ".join(f"{k} {r4.auc(y, f(t)):.3f}" for k, f in SIG.items())
        else:
            line += " too few failures"
        print(line)
    t = m[(m.dataset == ds) & m.p_ne]
    if t.failure.sum() > 2:
        tests(t, [("SegFormer", "YOLO")], SIG)

# ------------------------------------------------------------------ B. PolypDB
p = pd.read_csv(PDB); p["image"] = p.image.astype(str)
case = p.image.str.extract(r"^(.*_BN\d+)_\d+$")[0]
p["cl"] = np.where((p.centre == "BKAI") & case.notna(), case, p.centre + "_" + p.image)
p["failure"] = (p.dsc_primary < 0.5).astype(int); p["p_ne"] = p.px_primary > 0
PS = {"SegFormer": "agr_segformer", "YOLO": "agr_yolo", "UNet++": "agr_unetpp", "UNet++ matched": "agr_unetpp_m",
      "Confidence": "conf_det", "Morphology": "circularity"}
PS = {k: v for k, v in PS.items() if v in p.columns}
PSIG = {k: (lambda c: (lambda t: -t[c].values))(v) for k, v in PS.items()}
print("\nB. PolypDB external validation")
for gname, g in [("main (BKAI + Karolinska)", p[p.group == "main"]), ("sensitivity (Simula NBI)", p[p.group == "sensitivity"])]:
    print(f"  {gname}: n={len(g)}, failures={g.failure.sum()} ({g.failure.mean():.1%}), empty primary={int((~g.p_ne).sum())}, "
          f"mean primary DSC={g.dsc_primary.mean():.3f}, clusters={g.cl.nunique()}")
    for sub, t in [("full", g), ("non-empty", g[g.p_ne])]:
        y = t.failure.values
        if not 1 < y.sum() < len(y) - 1:
            print(f"    {sub}: too few failures"); continue
        print(f"    {sub} (n={len(t)}, failures={y.sum()}): " + "; ".join(
            f"{k} {r4.auc(y, f(t)):.3f} [{boot_ci(y, f(t))[0]:.3f}, {boot_ci(y, f(t))[1]:.3f}]" for k, f in PSIG.items()))
        tests(t, [("SegFormer", "YOLO"), ("SegFormer", "Confidence"), ("SegFormer", "Morphology")]
              + ([("UNet++ matched", "YOLO")] if "UNet++ matched" in PSIG else []), PSIG)
    ne = g[g.p_ne]; f = ne[ne.failure == 1]
    refs = [r for r in ["yolo", "segformer", "unetpp", "unetpp_m"] if f"dsc_{r}" in p.columns]
    print("    non-empty primary failures also failed by referee:", {r: f"{int((f['dsc_' + r] < 0.5).sum())}/{len(f)}" for r in refs})
    print("    referee mean DSC (non-empty):", {r: round(ne["dsc_" + r].mean(), 3) for r in refs})
g = p[p.group == "main"]
print("  By imaging modality (main set):")
for mod in ["WLI", "BLI", "FICE", "LCI"]:
    for sub, t in [("full", g[g.modality == mod]), ("non-empty", g[(g.modality == mod) & g.p_ne])]:
        y = t.failure.values
        s = f"    {mod:4s} {sub:9s} n={len(t):4d} failures={y.sum():3d} mean DSC={t.dsc_primary.mean():.3f} | "
        s += " ".join(f"{k} {r4.auc(y, f(t)):.3f}" for k, f in PSIG.items()) if 1 < y.sum() < len(y) - 1 else "too few failures"
        print(s)
x = g[g.near_duplicate_of.isna() | (g.near_duplicate_of.astype(str).str.strip() == "")]
print(f"  Excluding {len(g) - len(x)} images flagged as possible near-duplicates:")
for sub, t in [("full", x), ("non-empty", x[x.p_ne])]:
    y = t.failure.values
    print(f"    {sub} (n={len(t)}, failures={y.sum()}): " + " ".join(f"{k} {r4.auc(y, f(t)):.3f}" for k, f in PSIG.items()))
tests(x[x.p_ne], [("SegFormer", "YOLO"), ("SegFormer", "Confidence"), ("SegFormer", "Morphology")], PSIG)

# ------------------------------------------------------------------ C. condition strata
c = pd.read_csv(COND); c["image"] = c.image.astype(str)
mc = m.merge(c, on=["dataset", "image"])
print("\nC. Failure detection by tertile of lesion size and imaging condition (standardized benchmark)")
for var, lab in [("gt_frac", "polyp area fraction"), ("sharpness", "sharpness"), ("brightness", "brightness"),
                 ("specular_frac", "specular fraction")]:
    print(f"  {lab}")
    for sub, t in [("full", mc), ("non-empty", mc[mc.p_ne])]:
        q = pd.qcut(t[var], 3, labels=["low", "mid", "high"])
        for lv in ["low", "mid", "high"]:
            tt = t[q == lv]; y = tt.failure.values
            s = (f"    {sub:9s} {lv:4s} [{tt[var].min():.4g}, {tt[var].max():.4g}] n={len(tt)} failures={y.sum()} | ")
            s += " ".join(f"{k} {r4.auc(y, f(tt)):.3f}" for k, f in SIG.items() if k != "TTA") if 1 < y.sum() < len(y) - 1 else ""
            comp = tt.dataset.replace({"cvc_300": "cvc_colondb"}).value_counts(normalize=True)
            print(s + "  | sources: " + ", ".join(f"{k} {v:.0%}" for k, v in comp.items()))
