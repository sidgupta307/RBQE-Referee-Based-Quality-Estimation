#!/usr/bin/env python3
"""
Revised Figure 4: risk-coverage curves for SegFormer-B0 RBQE, Independent YOLO
RBQE and the primary model's own confidence, on the full benchmark (a) and on
the images with a non-empty primary mask (b). Ties are handled by averaging
over random orderings of tied images (expected mean retained DSC).

Usage: python R1_figure4_risk_coverage.py sample_results primary_confidence.csv
Output: Figure_4.pdf / Figure_4.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
res, conf = sys.argv[1], sys.argv[2]
m = pd.read_csv(f"{res}/merged_referee_scores.csv")
c = pd.read_csv(conf)
m["image"] = m.image.astype(str); c["image"] = c.image.astype(str)
d = m.merge(c, on=["dataset", "image"])
assert len(d) == 1223, len(d)

signals = [("RBQE, SegFormer-B0", "segformer_score", "#c62828", "-"),
           ("RBQE, Independent YOLO", "yolo_score", "#6a3d9a", "-"),
           ("Primary-model confidence", "conf_det", "#555555", "--")]
cov = np.arange(0.10, 1.0001, 0.01)


def curve(rel, dsc):
    out = []
    srt = np.sort(rel)[::-1]
    for cv in cov:
        k = int(round(cv * len(rel)))
        thr = srt[k - 1]
        above = dsc[rel > thr]; tie = dsc[rel == thr]
        out.append((above.sum() + tie.mean() * (k - len(above))) / k)
    return np.array(out)


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=False)
for ax, (title, mask) in zip(axes, (("(a) Full benchmark (n = 1,223)", np.ones(len(d), bool)),
                                    ("(b) Non-empty primary mask (n = 1,046)", (d.n_det > 0).values))):
    dsc = d.gt_dice.values[mask]
    for name, col, colour, ls in signals:
        ax.plot(cov, curve(d[col].values[mask], dsc), ls, color=colour, lw=2, label=name)
    ax.set_xlim(1.0, 0.08)
    ax.set_xlabel("Coverage (fraction of predictions retained)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Mean DSC of retained predictions")
axes[1].legend(loc="lower right", frameon=True)
fig.tight_layout(pad=0.4)
fig.savefig("Figure_4.pdf"); fig.savefig("Figure_4.png", dpi=600)
for title, mask in (("full", np.ones(len(d), bool)), ("non-empty", (d.n_det > 0).values)):
    dsc = d.gt_dice.values[mask]
    for name, col, *_ in signals:
        v = curve(d[col].values[mask], dsc)
        pick = {cv: v[np.argmin(abs(cov - cv))] for cv in (0.9, 0.8, 0.5, 0.1)}
        print(title, name, " ".join(f"{k:.1f}:{x:.3f}" for k, x in pick.items()))
