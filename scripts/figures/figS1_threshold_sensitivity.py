"""
Supplementary Figure S1 - Threshold sensitivity (Independent YOLO Referee, Agreement Dice)
Values: Supplementary Table S2, written directly at 3 decimals so labels match the text
(0.9105 must display as 0.911; formatting the 4-decimal float gives 0.910).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
auc        = [0.940, 0.937, 0.923, 0.917, 0.911]   # Table S2, rounded to 3 decimals

fig, ax = plt.subplots(figsize=(7, 4.6), dpi=300)
ax.plot(thresholds, auc, marker="o", color="C0", linewidth=1.8, markersize=5)
for t, a in zip(thresholds, auc):
    ax.annotate(f"{a:.3f}", (t, a), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8)
ax.set_xticks(thresholds)
ax.set_ylim(0.88, 0.96)
ax.set_xlabel("Failure threshold (DSC < threshold)")
ax.set_ylabel("ROC-AUC")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figS1_threshold_sensitivity.png", dpi=300, facecolor="white")
plt.savefig("figS1_threshold_sensitivity.pdf", facecolor="white")
print("Saved Figure S1")
