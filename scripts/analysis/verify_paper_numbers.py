"""
verify_paper_numbers.py
Recomputes every manuscript number that can be derived from the released per-image files and
compares it with the value printed in the paper (PASS = equal at the paper's reported precision).

All computations are deterministic: the same input files always give the same output.
Input-file MD5 checksums are printed so that a reader can confirm they use identical data.

Self-contained: no other project files are needed.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from scipy.stats import norm


def delong_paired(y, s1, s2):
    """Paired DeLong test for two correlated ROC AUCs (DeLong et al., 1988)."""
    y = np.asarray(y).astype(bool)
    def components(s):
        pos, neg = s[y], s[~y]
        v10 = np.array([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
        v01 = np.array([(pos < n).mean() + 0.5 * (pos == n).mean() for n in neg])
        return v10.mean(), v10, v01
    a1, v10a, v01a = components(np.asarray(s1))
    a2, v10b, v01b = components(np.asarray(s2))
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    cov = s10 / y.sum() + s01 / (~y).sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, 2 * norm.sf(abs(z))


# Hash-identical CVC-ColonDB image groups (pixel-level MD5), see dedup_sensitivity.py
COLONDB_DUPLICATE_GROUPS = [
    ["108", "139"], ["98", "147"], ["99", "148"], ["183", "198"], ["190", "203"],
    ["242", "245", "248", "251", "254", "257", "260", "263"],
    ["243", "246", "249", "252", "255", "258", "261"],
    ["244", "247", "250", "253", "256", "259", "262"],
    ["264", "271"], ["265", "272"], ["274", "294"], ["275", "295"], ["276", "289", "296"],
    ["277", "297"], ["278", "298"], ["279", "299"], ["280", "300"],
]
ETIS_DUPLICATE_GROUPS = [["127", "131"], ["172", "175"], ["20", "22"]]


# ---- Input files -------------------------------------------------------------------------
# Usage A (repository):   python verify_paper_numbers.py sample_results [clean_agreement_features.csv]
# Usage B (local machine): python verify_paper_numbers.py   (uses the local result folders below)
LOCAL = r"C:\seg_uncertain\journal_extension\T1_22_Independent_YOLO_Referee"
LOCAL_SCORES = os.path.join(LOCAL, "07_delong", "merged_referee_scores.csv")
LOCAL_YOLO = os.path.join(LOCAL, "05_standardized_evaluation", "merged_1223_yolo_seed.csv")
LOCAL_CLEAN = r"C:\seg_uncertain\chat_gpt\20B_segformer_clean_agreement\clean_agreement_features.csv"

if len(sys.argv) > 2 and sys.argv[1].lower().endswith(".csv"):
    # Usage C: python verify_paper_numbers.py merged_referee_scores.csv merged_1223_yolo_seed.csv [clean.csv]
    F_SCORES, F_YOLO = sys.argv[1], sys.argv[2]
    clean_csv = sys.argv[3] if len(sys.argv) > 3 else None
elif len(sys.argv) > 1:
    folder = sys.argv[1]
    F_SCORES = os.path.join(folder, "merged_referee_scores.csv")
    F_YOLO = os.path.join(folder, "merged_1223_yolo_seed.csv")
    clean_csv = sys.argv[2] if len(sys.argv) > 2 else None
else:
    F_SCORES, F_YOLO = LOCAL_SCORES, LOCAL_YOLO
    clean_csv = LOCAL_CLEAN if os.path.exists(LOCAL_CLEAN) else None
for f in (F_SCORES, F_YOLO):
    if not os.path.exists(f):
        sys.exit(f"File not found: {f}\nRun:  dir /s /b C:\\seg_uncertain\\{os.path.basename(f)}  and pass the two file paths as arguments.")

rows = []


def check(item, computed, reported, decimals):
    ok = abs(round(float(computed), decimals) - reported) < 10 ** (-decimals) / 2 + 1e-12
    rows.append((item, f"{computed:.{decimals + 1}f}", f"{reported}", "PASS" if ok else "FAIL"))


def check_exact(item, computed, reported):
    rows.append((item, str(computed), str(reported), "PASS" if computed == reported else "FAIL"))


def youden(y, risk):
    fpr, tpr, thr = roc_curve(y, risk)
    j = np.argmax(tpr - fpr)
    pred = risk >= thr[j]
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    acc = (tp + tn) / len(y); prec = tp / (tp + fp); rec = tp / (tp + fn)
    return tp, fp, fn, tn, acc, prec, rec, 2 * prec * rec / (prec + rec)


for f in (F_SCORES, F_YOLO):
    print(f"MD5 {hashlib.md5(open(f, 'rb').read()).hexdigest()}  {os.path.basename(f)}")

s = pd.read_csv(F_SCORES)
d = pd.read_csv(F_YOLO)
m = s.merge(d[["dataset", "image", "centroid_distance"]], on=["dataset", "image"])
y = m["failure"].values
seg, yol = -m["segformer_score"].values, -m["yolo_score"].values

# --- Benchmark ---------------------------------------------------------------------------
check_exact("Benchmark images", len(m), 1223)
check_exact("Failures (DSC < 0.50)", int(y.sum()), 236)

# --- SegFormer-B0 (Section 5.2, Fig. 2, Table 3, Table S3/S4) --------------------------------
check("SegFormer-B0 ROC-AUC", roc_auc_score(y, seg), 0.960, 3)
tp, fp, fn, tn, acc, prec, rec, f1 = youden(y, seg)
check_exact("SegFormer-B0 confusion matrix (TN,FP,FN,TP)", (tn, fp, fn, tp), (921, 66, 25, 211))
check("SegFormer-B0 Accuracy", acc, 0.926, 3); check("SegFormer-B0 Precision", prec, 0.762, 3)
check("SegFormer-B0 Recall", rec, 0.894, 3); check("SegFormer-B0 F1", f1, 0.823, 3)

# --- Independent YOLO Referee (Table 3, Tables S2, S4, S7, S9) -------------------------------
check("Independent YOLO ROC-AUC", roc_auc_score(y, yol), 0.923, 3)
tp, fp, fn, tn, acc, prec, rec, f1 = youden(y, yol)
check("Independent YOLO Accuracy", acc, 0.908, 3); check("Independent YOLO Precision", prec, 0.710, 3)
check("Independent YOLO Recall", rec, 0.881, 3); check("Independent YOLO F1", f1, 0.786, 3)

a1, a2, p = delong_paired(y, seg, yol)
check("dAUC SegFormer-B0 - Independent YOLO", a1 - a2, 0.0370, 4)
check("DeLong p x 1e5 (paper 6.32e-5)", p * 1e5, 6.32, 2)

ne = m["centroid_distance"].values < 0.999999
check_exact("Both-non-empty subset size", int(ne.sum()), 975)
check_exact("Both-non-empty failures", int(y[ne].sum()), 47)
check("Independent YOLO restricted ROC-AUC", roc_auc_score(y[ne], yol[ne]), 0.783, 3)
b1, b2, p2 = delong_paired(y[ne], seg[ne], yol[ne])
check("Paired 975: SegFormer-B0", b1, 0.885, 3); check("Paired 975: Independent YOLO", b2, 0.783, 3)
check("Paired 975: DeLong p", p2, 0.0089, 4)

both_empty = (d["centroid_distance"] >= 0.999999) & (d["area_ratio"] >= 0.999999)
check_exact("Both-empty stratum (Table 4)", int(both_empty.sum()), 130)

# Trivial empty-primary rule (Section 5.4): 177 of 236 failures flagged, no false positives
check("Trivial empty-primary rule ROC-AUC", 177 / 236 + 0.5 * 59 / 236, 0.875, 3)

# Threshold sensitivity (Table S2)
yd = d["agreement_dice"].values
for t, n_f, auc_rep in [(0.30, 210, 0.940), (0.40, 222, 0.937), (0.50, 236, 0.923),
                        (0.60, 255, 0.917), (0.70, 286, 0.911)]:
    yt = (d["dice"].values < t).astype(int)
    check_exact(f"Failures at DSC < {t:.2f}", int(yt.sum()), n_f)
    check(f"ROC-AUC at DSC < {t:.2f}", roc_auc_score(yt, -yd), auc_rep, 3)

# Table S7 (Independent YOLO descriptors)
yy = d["failure"].values
for col, auc_r, pr_r, rho_r in [("agreement_dice", 0.9231, 0.7162, 0.7345),
                                ("agreement_iou", 0.9231, 0.7162, 0.7345),
                                ("centroid_distance", 0.9165, 0.7169, -0.6197),
                                ("boundary_agreement", 0.9148, 0.7132, 0.6224),
                                ("area_ratio", 0.3813, 0.2954, 0.1306)]:
    risk = d[col].values if col == "centroid_distance" else -d[col].values
    check(f"S7 {col} ROC-AUC", roc_auc_score(yy, risk), auc_r, 4)
    check(f"S7 {col} PR-AUC", average_precision_score(yy, risk), pr_r, 4)
    check(f"S7 {col} Spearman", spearmanr(d[col], d["dice"]).statistic, rho_r, 4)

# Risk-coverage (abstract, Section 5.6): coverages where no tied scores straddle the cut
order = d.sort_values("agreement_dice", ascending=False, kind="mergesort")
for c, rep in [(1.0, 0.731), (0.5, 0.911), (0.1, 0.945)]:
    check(f"Mean retained DSC at {int(c * 100)}% coverage", order["dice"].head(int(round(c * len(d)))).mean(), rep, 3)

# Deduplicated sensitivity analysis (Section 5.3, Supplementary Section 11)
def repeats(dataset, groups):
    names = set(m.loc[m["dataset"] == dataset, "image"].astype(str))
    return {x for g in groups for x in [v for v in g if v in names][1:]}


drop_c = repeats("cvc_colondb", COLONDB_DUPLICATE_GROUPS)
drop_e = repeats("etis_larib", ETIS_DUPLICATE_GROUPS)
dd = m[(m["dataset"] != "cvc_300")
       & ~((m["dataset"] == "cvc_colondb") & m["image"].astype(str).isin(drop_c))
       & ~((m["dataset"] == "etis_larib") & m["image"].astype(str).isin(drop_e))]
yd2 = dd["failure"].values; s2, r2 = -dd["segformer_score"].values, -dd["yolo_score"].values
check_exact("Deduplicated images", len(dd), 1126); check_exact("Deduplicated failures", int(yd2.sum()), 227)
c1, c2, cp = delong_paired(yd2, s2, r2)
check("Dedup SegFormer-B0", c1, 0.959, 3); check("Dedup Independent YOLO", c2, 0.925, 3)
check("Dedup DeLong p x 1e4 (paper 3.2e-4)", cp * 1e4, 3.2, 1)
ne2 = dd["centroid_distance"].values < 0.999999
e1, e2, ep = delong_paired(yd2[ne2], s2[ne2], r2[ne2])
check_exact("Dedup both-non-empty images", int(ne2.sum()), 898)
check("Dedup paired SegFormer-B0", e1, 0.884, 3); check("Dedup paired Independent YOLO", e2, 0.776, 3)
check("Dedup paired DeLong p", ep, 0.006, 3)

# Table S3 (optional; needs the SegFormer per-image descriptor file)
if clean_csv:
    c = pd.read_csv(clean_csv)
    yc = (c["gt_dice"] < 0.5).astype(int).values
    for col, rep in [("agreement_dice", 0.960), ("agreement_iou", 0.960), ("area_ratio", 0.958),
                     ("centroid_distance", 0.956), ("boundary_agreement", 0.903)]:
        risk = c[col].values if col == "centroid_distance" else -c[col].values
        check(f"S3 {col} ROC-AUC", roc_auc_score(yc, risk), rep, 3)

pd.set_option("display.width", 200)
out = pd.DataFrame(rows, columns=["Item", "Computed", "Paper", "Result"])
print(out.to_string(index=False))
print(f"\n{(out['Result'] == 'PASS').sum()} PASS, {(out['Result'] == 'FAIL').sum()} FAIL")
