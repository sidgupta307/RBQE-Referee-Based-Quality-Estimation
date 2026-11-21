#!/usr/bin/env python3
"""
Reviewer 6, Comment 8 - deployment-oriented analysis of the reliability score.

Thresholds are chosen on the standardized benchmark (development data) and
applied unchanged to PolypDB (external data):
  * target failure sensitivity 90% and 95% (flag if score <= threshold)
  * fixed review budgets of 20% and 30%
  * Youden threshold
For each operating point: referral rate, failures caught (sensitivity),
specificity, PPV among referred, failure rate among accepted predictions
(residual risk), correct masks referred (false referrals).
Also: the same sensitivity targets for the Independent YOLO Referee and
primary-model confidence; a four-tier workflow (empty primary / review /
accept / accept with high agreement); high-agreement failures; and an
exploratory add-on rule that also refers low-confidence detections.

Usage:
  python R8_operating_points.py sample_results primary_confidence.csv polypdb_results.csv
Needs R4_sequence_level_analysis.py in the same folder.
"""
import sys
import numpy as np
import pandas as pd

import R4_sequence_level_analysis as r4

RES, CONF, PDB = sys.argv[1:4]
r4.RES, r4.CONF = RES, CONF
d = r4.load()
dev = pd.DataFrame(dict(sf=d.segformer_score, yolo=d.yolo_score, conf=d.conf_det, y=d.failure,
                        is_empty=d.n_det == 0, dsc=d.gt_dice))
p = pd.read_csv(PDB); p = p[p.group == "main"]
ext = pd.DataFrame(dict(sf=p.agr_segformer, yolo=p.agr_yolo, conf=p.conf_det, y=(p.dsc_primary < 0.5).astype(int),
                        is_empty=p.px_primary == 0, dsc=p.dsc_primary))
DATA = {"Benchmark": dev, "PolypDB": ext}


def summary(t, col, tau):
    f = t[col] <= tau; y = t.y == 1
    tp, fn, fp, tn = (f & y).sum(), (~f & y).sum(), (f & ~y).sum(), (~f & ~y).sum()
    return (f"referred {f.mean():5.1%} ({f.sum():4d}) | sensitivity {tp / (tp + fn):.3f} ({tp}/{tp + fn}) | "
            f"specificity {tn / (tn + fp):.3f} | PPV {tp / max(1, tp + fp):.3f} | "
            f"failure rate among accepted {fn / max(1, (~f).sum()):.2%} ({fn}/{(~f).sum()}) | correct masks referred {fp}")


def youden(s, y):
    th = np.unique(s)
    return max(th, key=lambda t: ((s <= t) & (y == 1)).sum() / y.sum() + ((s > t) & (y == 0)).sum() / (y == 0).sum())


s, y = dev.sf.values, dev.y.values
ops = {"90% sensitivity": np.quantile(s[y == 1], 0.90, method="higher"),
       "95% sensitivity": np.quantile(s[y == 1], 0.95, method="higher"),
       "20% review budget": np.quantile(s, 0.20), "30% review budget": np.quantile(s, 0.30),
       "Youden": youden(s, y)}
print("1. SegFormer-B0 agreement: thresholds chosen on the benchmark, applied unchanged to PolypDB")
print(f"   Without screening, failure rate = {dev.y.mean():.1%} (benchmark), {ext.y.mean():.1%} (PolypDB)")
for k, tau in ops.items():
    print(f"  {k} (threshold {tau:.3f})")
    for nm, t in DATA.items():
        print(f"    {nm:9s} {summary(t, 'sf', tau)}")

print("\n2. Same sensitivity targets for each signal (threshold chosen on the benchmark)")
for target in (0.90, 0.95):
    for lab, col in [("SegFormer-B0", "sf"), ("Independent YOLO", "yolo"), ("Primary confidence", "conf")]:
        tau = np.quantile(dev[col].values[y == 1], target, method="higher")
        for nm, t in DATA.items():
            print(f"  {target:.0%} {lab:18s} threshold {tau:.3f} {nm:9s} {summary(t, col, tau)}")

TAU = ops["90% sensitivity"]
print(f"\n3. Four-tier workflow (review threshold = 90%-sensitivity threshold {TAU:.3f}; high agreement > 0.9)")
for nm, t in DATA.items():
    tiers = {"Tier 1 empty primary mask (automatic referral)": t.is_empty,
             "Tier 2 agreement <= threshold (expert review)": ~t.is_empty & (t.sf <= TAU),
             "Tier 3 threshold < agreement <= 0.9 (accept)": ~t.is_empty & (t.sf > TAU) & (t.sf <= 0.9),
             "Tier 4 agreement > 0.9 (accept, high agreement)": ~t.is_empty & (t.sf > 0.9)}
    print(f"  {nm} (n = {len(t)}, failures = {t.y.sum()})")
    for k, v in tiers.items():
        print(f"    {k:50s} images {v.sum():4d} ({v.mean():5.1%})  failures {t.y[v].sum():3d} ({t.y[v].mean():.1%} of tier)")

print("\n4. Accepted failures at the 90%-sensitivity threshold (high-agreement failures)")
for nm, t in DATA.items():
    a = t[~t.is_empty & (t.sf > TAU) & (t.y == 1)]
    print(f"  {nm:9s} accepted failures {len(a)}: DSC median {a.dsc.median():.2f}, range {a.dsc.min():.2f}-{a.dsc.max():.2f}; "
          f"DSC < 0.1: {(a.dsc < 0.1).sum()} (of which primary confidence < 0.40: {((a.dsc < 0.1) & (a.conf < 0.40)).sum()})")

print("\n5. Exploratory add-on: also refer detections with primary confidence below c")
for c in (0.30, 0.35, 0.40, 0.50):
    for nm, t in DATA.items():
        base = t.sf <= TAU; comb = base | (t.conf < c); yy = t.y == 1; extra = comb & ~base
        print(f"  c = {c:.2f} {nm:9s} referred {base.mean():.1%} -> {comb.mean():.1%} (+{extra.sum()}); "
              f"sensitivity {(base & yy).sum() / yy.sum():.3f} -> {(comb & yy).sum() / yy.sum():.3f} (+{(extra & yy).sum()}); "
              f"failure rate among accepted {(~base & yy).sum() / (~base).sum():.2%} -> {(~comb & yy).sum() / (~comb).sum():.2%}")
