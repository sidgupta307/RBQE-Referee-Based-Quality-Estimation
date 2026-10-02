"""
Overlap/duplicate sensitivity analysis (reported in Section 5.3 and Supplementary Section 11).

Pixel-level MD5 hashing of the dataset images showed:
  * all 60 CVC-300 images are pixel-identical to CVC-ColonDB images 149-208
    (59 with identical ground-truth masks);
  * CVC-ColonDB's 380 image files contain 346 unique images (17 groups, 34 repeated files);
  * ETIS-Larib's 196 image files contain 193 unique images (3 groups, 3 repeated files);
  * CVC-ClinicDB (612) and Kvasir-SEG (1,000) contain no repeats, and Kvasir-SEG shares no image
    with any external set.

This script removes CVC-300 and the repeated CVC-ColonDB and ETIS-Larib files (keeping the first copy of each
group) and recomputes the referee comparison on the deduplicated benchmark.

Usage:
    python scripts/analysis/dedup_sensitivity.py sample_results
"""
import os
import sys
import pandas as pd
from paired_delong_and_restricted import delong_paired   # same folder

# Hash-identical CVC-ColonDB image groups (file stems), from pixel-level MD5 hashing
COLONDB_DUPLICATE_GROUPS = [
    ["108", "139"], ["98", "147"], ["99", "148"], ["183", "198"], ["190", "203"],
    ["242", "245", "248", "251", "254", "257", "260", "263"],
    ["243", "246", "249", "252", "255", "258", "261"],
    ["244", "247", "250", "253", "256", "259", "262"],
    ["264", "271"], ["265", "272"], ["274", "294"], ["275", "295"], ["276", "289", "296"],
    ["277", "297"], ["278", "298"], ["279", "299"], ["280", "300"],
]
ETIS_DUPLICATE_GROUPS = [["127", "131"], ["172", "175"], ["20", "22"]]

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample_results"
    a = pd.read_csv(os.path.join(folder, "merged_referee_scores.csv"))
    b = pd.read_csv(os.path.join(folder, "merged_1223_yolo_seed.csv"))
    m = a.merge(b, on=["dataset", "image"], suffixes=("", "_y"))


    def report(df, label):
        y = df["failure"].values
        s, r = -df["segformer_score"].values, -df["yolo_score"].values
        a1, a2, p = delong_paired(y, s, r)
        ne = df["centroid_distance"].values < 0.999999
        b1, b2, p2 = delong_paired(y[ne], s[ne], r[ne])
        print(f"{label:38s} n={len(df):5d} failures={y.sum():3d} | SegFormer-B0 {a1:.3f}  Independent YOLO {a2:.3f}  "
              f"DeLong p={p:.1e} | both non-empty n={ne.sum()} (failures {y[ne].sum()}): {b1:.3f} vs {b2:.3f}  p={p2:.4f}")


    report(m, "Standardized benchmark (as reported)")
    def repeats(dataset, groups):
        names = set(m.loc[m["dataset"] == dataset, "image"].astype(str))
        return {x for g in groups for x in [v for v in g if v in names][1:]}

    drop_c = repeats("cvc_colondb", COLONDB_DUPLICATE_GROUPS)
    drop_e = repeats("etis_larib", ETIS_DUPLICATE_GROUPS)
    dedup = m[(m["dataset"] != "cvc_300")
              & ~((m["dataset"] == "cvc_colondb") & m["image"].astype(str).isin(drop_c))
              & ~((m["dataset"] == "etis_larib") & m["image"].astype(str).isin(drop_e))]
    report(dedup, "Deduplicated benchmark")
