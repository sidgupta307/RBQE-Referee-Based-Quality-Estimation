"""
Supplementary Figure S4 - Forward-pass burden vs. ROC-AUC (structural, not latency)
Values: Supplementary Table S10. All RBQE points use the COMMON Agreement Dice descriptor
(UNet++ = 0.938, MedSAM = 0.863), NOT the best-descriptor values from Table S6.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

points = [  # (label, forward passes, ROC-AUC, colour)
    ("Morphology-based\nSQE",       1, 0.927, "C0"),
    ("TTA",                         9, 0.905, "C1"),
    ("RBQE\nIndependent YOLO",      2, 0.923, "C2"),
    ("RBQE\nSegFormer-B0",          2, 0.960, "C3"),
    ("RBQE\nUNet++",                2, 0.938, "C4"),
    ("RBQE\nMedSAM",                2, 0.863, "C5"),
]
# label offsets (points) chosen so no two labels overlap
offsets = {"RBQE\nIndependent YOLO": (8, -14), "Morphology-based\nSQE": (8, 4)}

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
for label, x, y, c in points:
    ax.scatter(x, y, s=60, color=c, zorder=3)
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=offsets.get(label, (8, 4)), fontsize=7.5, va="bottom")
ax.set_xticks(range(1, 10))
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0.84, 0.98)
ax.set_xlabel("Total model forward passes per image")
ax.set_ylabel("Reported failure-detection ROC-AUC")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figS4_forward_pass_burden.png", dpi=300, facecolor="white")
plt.savefig("figS4_forward_pass_burden.pdf", facecolor="white")
print("Saved Figure S4")
