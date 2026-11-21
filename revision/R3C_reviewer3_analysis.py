#!/usr/bin/env python3
"""
Reviewer 3 - reproduces every number in the response to Reviewer 3.
  A. Comment 2: all signals on the full, non-curated 1,248-image set
  B. Comment 1: Independent YOLO Referee retrained at 512 x 512 (R3A/R3B output)
     on the benchmark and on PolypDB, with shared failures and referee DSC
  C. Comment 3: high-agreement failures (Agreement Dice > 0.7/0.8/0.9) per
     referee configuration, with accepted fractions; equal-sensitivity referrals
  D. Comment 6: empty-mask rule on the non-empty subset
  E. Comments 5 and 11: head-to-head on the common subset (primary, SegFormer-B0
     and Independent YOLO masks non-empty; 964 images), all referees

Usage:
  python R3C_reviewer3_analysis.py sample_results primary_confidence.csv referee_quality.csv ^
      yolo512_referee.csv agreement_features_independent.csv morphology_features.csv ^
      tta_uncertainty_features.csv polypdb_results.csv polypdb_yolo512.csv
Needs R4_sequence_level_analysis.py in the same folder.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import R4_sequence_level_analysis as r4

RES, CONF, QUAL, Y512, MEDSAM, MORPH, TTA, PDB, PDB512 = sys.argv[1:10]
r4.RES, r4.CONF = RES, CONF
B, SEED = 2000, 2026


def boot(y, s):
    rng = np.random.default_rng(SEED); a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a.append(r4.auc(y[i], s[i]))
    return np.percentile(a, [2.5, 97.5])


def fmt(y, s):
    lo, hi = boot(y, s); return f"{r4.auc(y, s):.3f} [{lo:.3f}, {hi:.3f}]"


def compare(t, a, b, sig, label=""):
    y = t.failure.values.astype(int); cl = t.cl.values; s1, s2 = sig[a](t), sig[b](t)
    _, _, D, _, p, _ = r4.clustered_delong(y, s1, s2, cl); lo, hi = r4.cluster_boot(y, s1, s2, cl)
    print(f"    {label}{a} vs {b}: dAUC {D:+.3f}  p_img {r4.image_delong(y, s1, s2):.4f}  p_cluster {p:.4f}  "
          f"p_jk {r4.cluster_jackknife_p(y, s1, s2, cl):.4f}  clusterCI [{lo:+.3f}, {hi:+.3f}]")


def norm(df):
    df = df.copy(); df["image"] = df.image.astype(str).str.replace(r"\.[A-Za-z]+$", "", regex=True); return df


q = norm(pd.read_csv(QUAL)); y5 = norm(pd.read_csv(Y512)); ms = pd.read_csv(MEDSAM)
ms["image"] = ms.image_key.astype(str) if "image_key" in ms.columns else ms.image.astype(str).str.replace(r"\.[A-Za-z]+$", "", regex=True)
mo = norm(pd.read_csv(MORPH)); ta = norm(pd.read_csv(TTA)); c = norm(pd.read_csv(CONF))
pub = norm(pd.read_csv(f"{RES}/merged_referee_scores.csv"))

# ------------------------------------------------------------------ A. full 1,248
f = (q.merge(c[["dataset", "image", "n_det", "conf_det"]], on=["dataset", "image"])
      .merge(y5[["dataset", "image", "agr_yolo512"]], on=["dataset", "image"])
      .merge(ms[["dataset", "image", "agreement_dice"]].rename(columns={"agreement_dice": "agr_medsam"}), on=["dataset", "image"])
      .merge(mo[["dataset", "image", "circularity"]], on=["dataset", "image"])
      .merge(ta[["dataset", "image", "num_zero_predictions"]], on=["dataset", "image"])
      .merge(pub[["dataset", "image", "segformer_score", "yolo_score"]], on=["dataset", "image"], how="left"))
assert len(f) == 1248
ex = f[f.segformer_score.isna()]
print(f"A. Full 1,248-image set. Excluded from the standardized benchmark: {len(ex)} images; "
      f"all with empty primary mask: {bool((ex.n_det == 0).all())}; failures among them: {int((ex.dsc_primary < 0.5).sum())}")
f["segformer_score"] = f.segformer_score.fillna(0.0); f["yolo_score"] = f.yolo_score.fillna(0.0)  # empty primary -> agreement 0
f["failure"] = (f.dsc_primary < 0.5).astype(int)
y = f.failure.values
FS = {"SegFormer-B0": -f.segformer_score.values, "Indep. YOLO (640)": -f.yolo_score.values, "Indep. YOLO (512)": -f.agr_yolo512.values,
      "UNet++": -f.agr_unetpp.values, "UNet++ matched": -f.agr_unetpp_m.values, "MedSAM": -f.agr_medsam.values,
      "Confidence": -f.conf_det.values, "Morphology": -f.circularity.values, "TTA": f.num_zero_predictions.values.astype(float),
      "Empty-mask rule": (f.n_det == 0).astype(float).values}
print(f"  n = {len(f)}, failures = {y.sum()}")
for k, s in FS.items():
    print(f"  {k:18s} {fmt(y, s)}")
for b in ["Indep. YOLO (640)", "Indep. YOLO (512)", "UNet++", "Confidence", "Morphology", "TTA"]:
    print(f"    SegFormer-B0 vs {b}: dAUC {r4.auc(y, FS['SegFormer-B0']) - r4.auc(y, FS[b]):+.3f}  p_img {r4.image_delong(y, FS['SegFormer-B0'], FS[b]):.2e}")

# ------------------------------------------------------------------ benchmark (1,223) with all referees
d = r4.assign_clusters(r4.load(), 15, 34)
m = (d.merge(q[["dataset", "image", "agr_unetpp", "agr_unetpp_m", "dsc_yolo", "dsc_segformer", "dsc_unetpp_m"]], on=["dataset", "image"])
      .merge(y5[["dataset", "image", "agr_yolo512", "dsc_yolo512", "px_yolo512"]], on=["dataset", "image"])
      .merge(ms[["dataset", "image", "agreement_dice"]].rename(columns={"agreement_dice": "agr_medsam"}), on=["dataset", "image"]))
assert len(m) == 1223
REF = {"SegFormer-B0": "segformer_score", "Indep. YOLO (640)": "yolo_score", "Indep. YOLO (512)": "agr_yolo512",
       "UNet++": "agr_unetpp", "UNet++ matched": "agr_unetpp_m", "MedSAM": "agr_medsam"}
SIG = {k: (lambda col: (lambda t: -t[col].values))(v) for k, v in REF.items()}
SIG["Confidence"] = lambda t: -t.conf_det.values

# ------------------------------------------------------------------ B. YOLO at 512
print("\nB. Independent YOLO Referee trained at 512 x 512 (benchmark)")
ne = m[m.n_det > 0]
for nm, t in [("full (1,223)", m), ("non-empty (1,046)", ne)]:
    yy = t.failure.values
    print(f"  {nm}: " + "; ".join(f"{k} {fmt(yy, SIG[k](t))}" for k in ["SegFormer-B0", "Indep. YOLO (640)", "Indep. YOLO (512)", "UNet++ matched", "MedSAM", "Confidence"]))
    for a, b in [("Indep. YOLO (512)", "Indep. YOLO (640)"), ("SegFormer-B0", "Indep. YOLO (512)"), ("UNet++ matched", "Indep. YOLO (512)")]:
        compare(t, a, b, SIG)
fl = ne[ne.failure == 1]
DSC = {"Indep. YOLO (640)": "dsc_yolo", "Indep. YOLO (512)": "dsc_yolo512", "SegFormer-B0": "dsc_segformer", "UNet++ matched": "dsc_unetpp_m"}
for k, col in DSC.items():
    print(f"  {k:18s} shared failures {int((fl[col] < 0.5).sum())}/{len(fl)}  referee mean DSC (non-empty) {ne[col].mean():.3f}  "
          f"Spearman rho with primary DSC {spearmanr(ne.gt_dice, ne[col])[0]:.2f}")

p = pd.read_csv(PDB); p["image"] = p.image.astype(str)
p5 = pd.read_csv(PDB512); p5["image"] = p5.image.astype(str)
p = p.merge(p5[["group", "centre", "modality", "image", "agr_yolo512", "dsc_yolo512"]], on=["group", "centre", "modality", "image"])
case = p.image.str.extract(r"^(.*_BN\d+)_\d+$")[0]
p["cl"] = np.where((p.centre == "BKAI") & case.notna(), case, p.centre + "_" + p.image)
p["failure"] = (p.dsc_primary < 0.5).astype(int); p["p_ne"] = p.px_primary > 0
PS = {"SegFormer-B0": lambda t: -t.agr_segformer.values, "Indep. YOLO (640)": lambda t: -t.agr_yolo.values,
      "Indep. YOLO (512)": lambda t: -t.agr_yolo512.values, "UNet++ matched": lambda t: -t.agr_unetpp_m.values,
      "Confidence": lambda t: -t.conf_det.values}
g = p[p.group == "main"]
print("  PolypDB main set")
for nm, t in [("full (1,230)", g), ("non-empty (1,104)", g[g.p_ne])]:
    yy = t.failure.values
    print(f"  {nm}: " + "; ".join(f"{k} {fmt(yy, s(t))}" for k, s in PS.items()))
    for a, b in [("SegFormer-B0", "Indep. YOLO (512)"), ("Indep. YOLO (512)", "Indep. YOLO (640)"), ("UNet++ matched", "Indep. YOLO (512)")]:
        compare(t, a, b, PS)
gne = g[g.p_ne]; gf = gne[gne.failure == 1]
for k, col in DSC.items():
    print(f"  {k:18s} shared failures {int((gf[col] < 0.5).sum())}/{len(gf)}  referee mean DSC {gne[col].mean():.3f}")

# ------------------------------------------------------------------ C. high-agreement failures
print("\nC. High-agreement failures (Agreement Dice > t), benchmark (236 failures; 59 with non-empty primary mask)")
y = m.failure.values
for k, col in REF.items():
    s = m[col].values; line = f"  {k:18s}"
    for thr in (0.7, 0.8, 0.9):
        hi = s > thr
        line += (f" | >{thr}: accepted {hi.mean():5.1%}, failures accepted {int((hi & (y == 1)).sum()):2d} "
                 f"({(hi & (y == 1)).sum() / y.sum():.1%} of all, {(hi & (y == 1)).sum() / 59:.1%} of non-empty), "
                 f"failure rate among accepted {(hi & (y == 1)).sum() / max(1, hi.sum()):.2%}")
    print(line)
print("  PolypDB main set (175 failures; 49 with non-empty primary mask), Agreement Dice > 0.8")
yy = g.failure.values
for k, col in [("SegFormer-B0", "agr_segformer"), ("Indep. YOLO (640)", "agr_yolo"), ("Indep. YOLO (512)", "agr_yolo512"),
               ("UNet++", "agr_unetpp"), ("UNet++ matched", "agr_unetpp_m")]:
    hi = g[col].values > 0.8
    print(f"  {k:18s} accepted {hi.mean():5.1%}, failures accepted {int((hi & (yy == 1)).sum())} "
          f"({(hi & (yy == 1)).sum() / yy.sum():.1%} of all, {(hi & (yy == 1)).sum() / 49:.1%} of non-empty), "
          f"failure rate among accepted {(hi & (yy == 1)).sum() / max(1, hi.sum()):.2%}")
print("  Equal failure sensitivity (90%, threshold chosen per referee on the benchmark)")
for k, col in REF.items():
    s = m[col].values; tau = np.quantile(s[y == 1], 0.90, method="higher"); fl2 = s <= tau
    print(f"  {k:18s} threshold {tau:.3f}  referred {fl2.mean():.1%}  correct masks referred {int((fl2 & (y == 0)).sum())}")

# ------------------------------------------------------------------ D. empty-mask rule
t = m[m.n_det > 0]; yy = t.failure.values
print(f"\nD. Empty-mask rule on the 1,046 non-empty images: ROC-AUC {r4.auc(yy, np.zeros(len(yy))):.3f} "
      f"(SegFormer-B0 {r4.auc(yy, -t.segformer_score.values):.3f}; confidence {r4.auc(yy, -t.conf_det.values):.3f})")

# ------------------------------------------------------------------ E. common subset
t = m[(m.n_det > 0) & m.sf_ne & m.yo_ne]; yy = t.failure.values
print(f"\nE. Common subset (primary, SegFormer-B0 and Independent YOLO masks non-empty): n = {len(t)}, failures = {yy.sum()}")
print("  " + "; ".join(f"{k} {fmt(yy, s(t))}" for k, s in SIG.items()))
for a, b in [("SegFormer-B0", "Indep. YOLO (640)"), ("UNet++ matched", "Indep. YOLO (640)"), ("UNet++", "Indep. YOLO (640)"),
             ("Indep. YOLO (512)", "Indep. YOLO (640)"), ("SegFormer-B0", "Indep. YOLO (512)"), ("SegFormer-B0", "UNet++ matched"),
             ("SegFormer-B0", "Confidence")]:
    compare(t, a, b, SIG)
