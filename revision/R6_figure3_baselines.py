#!/usr/bin/env python3
"""
Revised Figure 3 (Reviewer 6, Comment 6): ROC-AUC with 95% bootstrap CIs for
the no-reference baselines and RBQE on (a) the full standardized benchmark
(n = 1,223) and (b) the 1,046 images with a non-empty primary mask.
Values are the verified outputs of R1_evaluate_confidence_baseline.py and
R6_baseline_paired_tests.py (revised Table 7).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 12})

methods = ["Morphology-\nbased SQE", "Test-time\naugmentation", "Primary-model\nconfidence",
           "RBQE\nIndep. YOLO", "RBQE\nSegFormer-B0"]
colors = ["#0f3d66", "#1c7da8", "#7f7f7f", "#6a3d9a", "#c62828"]
panels = {
    "(a) Full benchmark (n = 1,223)":
        [(0.927, 0.900, 0.951), (0.905, 0.878, 0.929), (0.945, 0.926, 0.962),
         (0.923, 0.902, 0.942), (0.960, 0.947, 0.972)],
    "(b) Non-empty primary mask (n = 1,046)":
        [(0.709, 0.627, 0.784), (0.621, 0.562, 0.687), (0.779, 0.721, 0.832),
         (0.787, 0.728, 0.845), (0.876, 0.835, 0.911)],
}

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7), sharey=True)
for ax, (title, vals) in zip(axes, panels.items()):
    x = range(len(vals))
    v = [a for a, _, _ in vals]
    err = [[a - lo for a, lo, _ in vals], [hi - a for a, _, hi in vals]]
    ax.bar(x, v, width=0.62, color=colors, zorder=3)
    ax.errorbar(x, v, yerr=err, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
    for i, (a, lo, hi) in enumerate(vals):
        ax.text(i, hi + 0.015, "%.3f" % a, ha="center", va="bottom", fontsize=11.5,
                fontweight="bold" if i == len(vals) - 1 else "normal")
    ax.axhline(0.5, ls=":", lw=1.2, color="grey", zorder=2)
    ax.set_xticks(list(x)); ax.set_xticklabels(methods)
    ax.set_xlim(-0.5, len(vals) - 0.5); ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, loc="left")
    ax.grid(axis="y", ls="--", lw=0.8, alpha=0.3, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("ROC-AUC (failure detection)")
fig.tight_layout(pad=0.5, w_pad=1.5)
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(os.path.dirname(here), "Figure_3")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=600)
print("Saved", out + ".pdf/.png")
