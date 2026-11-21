#!/usr/bin/env python3
"""
Evaluate the primary model's own confidence as a failure-detection baseline and
compare it with RBQE (Agreement Dice) on the same images.

Inputs : primary_confidence.csv (from R1_extract_primary_confidence.py)
         sample_results/merged_referee_scores.csv (repository)
Outputs: printed tables + confidence_baseline_results.csv + risk_coverage_confidence.png

Usage:  python R1_evaluate_confidence_baseline.py primary_confidence.csv sample_results
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

conf_csv = sys.argv[1] if len(sys.argv) > 1 else "primary_confidence.csv"
res_dir = sys.argv[2] if len(sys.argv) > 2 else "sample_results"
B = 2000
rng = np.random.default_rng(2026)


def auc(score, y):                       # higher score = higher failure risk
    r = rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def delong(y, s1, s2):
    y = y.astype(bool)
    def comp(s):
        pos, neg = s[y], s[~y]
        v10 = np.array([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
        v01 = np.array([(pos < n).mean() + 0.5 * (pos == n).mean() for n in neg])
        return v10.mean(), v10, v01
    a1, x1, z1 = comp(s1); a2, x2, z2 = comp(s2)
    c = np.cov(np.vstack([x1, x2])) / y.sum() + np.cov(np.vstack([z1, z2])) / (~y).sum()
    z = (a1 - a2) / np.sqrt(c[0, 0] + c[1, 1] - 2 * c[0, 1])
    return a1, a2, 2 * norm.sf(abs(z))


def boot(y, s1, s2):
    d, a = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a1 = auc(s1[i], y[i]); a.append(a1); d.append(a1 - auc(s2[i], y[i]))
    return np.percentile(a, [2.5, 97.5]), np.percentile(d, [2.5, 97.5])


c = pd.read_csv(conf_csv)
m = pd.read_csv(f"{res_dir}/merged_referee_scores.csv")
c["image"] = c["image"].astype(str); m["image"] = m["image"].astype(str)
d = m.merge(c, on=["dataset", "image"], how="left")
miss = d["conf_det"].isna().sum()
print(f"Merged {len(d)} benchmark images; missing confidence rows: {miss}")
d = d.dropna(subset=["conf_det"])
print("Alignment check, |recomputed DSC - reported DSC| > 0.01:",
      int((abs(d.dsc_check - d.gt_dice) > 0.01).sum()), "images")
print("Empty primary predictions (n_det == 0):", int((d.n_det == 0).sum()), "(paper: 177)")

y = d["failure"].values.astype(int)
signals = {
    "RBQE SegFormer-B0 (Agreement Dice)": -d["segformer_score"].values,
    "RBQE Independent YOLO (Agreement Dice)": -d["yolo_score"].values,
    "Primary confidence, top detection": -d["conf_det"].values,
    "Primary confidence, max candidate": -d["conf_max"].values,
}
ref = signals["RBQE SegFormer-B0 (Agreement Dice)"]
rows = []
for subset, mask in (("Full benchmark", np.ones(len(d), bool)),
                     ("Non-empty primary", (d.n_det > 0).values)):
    yy = y[mask]
    print(f"\n{subset}: n = {mask.sum()}, failures = {yy.sum()}")
    for name, s in signals.items():
        a = auc(s[mask], yy)
        ci, dci = boot(yy, s[mask], ref[mask])
        if s is ref:
            print(f"  {name:42s} AUC {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            rows.append((subset, name, a, *ci, np.nan, np.nan, np.nan, np.nan))
        else:
            a_ref, a_s, p = delong(yy, ref[mask], s[mask])
            dd = a_ref - a_s
            dlo, dhi = -dci[1], -dci[0]          # CI of (SegFormer - signal)
            print(f"  {name:42s} AUC {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]   "
                  f"SegFormer minus this = {dd:+.3f} [{dlo:+.3f}, {dhi:+.3f}], DeLong p = {p:.3g}")
            rows.append((subset, name, a, *ci, dd, dlo, dhi, p))
pd.DataFrame(rows, columns=["subset", "signal", "auc", "auc_lo", "auc_hi", "dAUC_vs_segformer",
                            "d_lo", "d_hi", "delong_p"]).to_csv("confidence_baseline_results.csv", index=False)

# risk-coverage: mean retained DSC when ranking from most to least reliable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cov = np.linspace(0.1, 1.0, 10)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (subset, mask) in zip(axes, (("Full benchmark", np.ones(len(d), bool)),
                                     ("Non-empty primary", (d.n_det > 0).values))):
    dsc = d["gt_dice"].values[mask]
    print(f"\nRisk-coverage, {subset} (mean retained DSC, random tie order averaged):")
    for name, s in signals.items():
        rel = -s[mask]                                   # higher = more reliable
        vals = []
        for cv in cov:
            k = int(round(cv * len(rel)))
            thr = np.sort(rel)[::-1][k - 1]
            above = dsc[rel > thr]; tie = dsc[rel == thr]
            vals.append((above.sum() + tie.mean() * (k - len(above))) / k)
        ax.plot(cov, vals, "-o", ms=3, label=name)
        print(f"  {name:42s} " + " ".join(f"{v:.3f}" for v in vals[::-1]))
    ax.set_xlim(1.02, 0.08); ax.set_xlabel("Coverage"); ax.set_ylabel("Mean retained DSC")
    ax.set_title(subset); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("risk_coverage_confidence.png", dpi=200)
print("\nCoverage order printed: 1.0, 0.9, ..., 0.1")
print("Saved confidence_baseline_results.csv and risk_coverage_confidence.png")
