#!/usr/bin/env python3
"""
Reviewer 3, Comment 7 - perceptual-hash near-duplicate check on the standardized
1,223-image benchmark.

Uses the same difference hash as the PolypDB check (R7B_polypdb_external.py:
9x8 grey image, dHash, Hamming distance <= 4). Images whose hashes are within
the threshold are linked, linked images form groups, and one image per group
(the first file in name order) is kept. Failure detection is then recomputed on
the reduced benchmark (full and non-empty-primary), with paired DeLong tests.

Usage (Windows, from C:\\seg_uncertain; no GPU needed):
  python R11_benchmark_near_duplicates.py
  (optional: --res sample_results --conf primary_confidence.csv --max_hamming 4)
Needs R4_sequence_level_analysis.py in the same folder.
Outputs: benchmark_near_duplicates.csv and the results printed on screen.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2

import R4_sequence_level_analysis as r4

p = argparse.ArgumentParser()
p.add_argument("--res", default="sample_results")
p.add_argument("--conf", default="primary_confidence.csv")
p.add_argument("--max_hamming", type=int, default=4)
a = p.parse_args()
DATASETS = {
    "cvc_clinicdb": r"C:\Polygon_polyp_new(05_26)\cvc_dataset\images",
    "cvc_colondb": r"C:\Polygon_polyp_new(05_26)\cvc-colonDB\images",
    "etis_larib": r"C:\Polygon_polyp_new(05_26)\etis-larib\images",
    "cvc_300": r"C:\Polygon_polyp_new(05_26)\CVC-300\images",
}


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


r4.RES, r4.CONF = a.res, a.conf
d = r4.load().reset_index(drop=True)                       # the 1,223 benchmark images
hashes = []
for _, row in d.iterrows():
    folder = DATASETS[row.dataset]
    f = next((x for x in os.listdir(folder) if os.path.splitext(x)[0] == row.image), None)
    if f is None:
        raise SystemExit(f"Image not found: {row.dataset}/{row.image}")
    hashes.append(dhash(cv2.imread(os.path.join(folder, f))))
d["dhash"] = hashes

# link images within the Hamming threshold (union-find)
par = list(range(len(d)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
H = np.array(hashes, dtype=np.uint64)
pairs = 0
for i in range(len(d)):
    x = np.bitwise_xor(H[i + 1:], H[i])
    ham = np.array([bin(int(v)).count("1") for v in x])
    for j in np.where(ham <= a.max_hamming)[0] + i + 1:
        pairs += 1; par[find(i)] = find(int(j))
d["group"] = [find(i) for i in range(len(d))]
d = d.sort_values(["group", "dataset", "image"])
d["keep"] = ~d.duplicated("group")
d[["dataset", "image", "dhash", "group", "keep"]].to_csv("benchmark_near_duplicates.csv", index=False)
n_groups = (d.groupby("group").size() > 1).sum()
print(f"Benchmark images: {len(d)}; near-duplicate pairs (Hamming <= {a.max_hamming}): {pairs}; "
      f"groups with >1 image: {n_groups}; images removed: {(~d.keep).sum()} "
      f"(of which CVC-300: {((~d.keep) & (d.dataset == 'cvc_300')).sum()})")
k = d[d.keep]
SIG = {"SegFormer-B0": "segformer_score", "Independent YOLO": "yolo_score", "Confidence": "conf_det"}
for name, t in [("full", k), ("non-empty primary", k[k.p_ne])]:
    y = t.failure.values.astype(int)
    print(f"\nReduced benchmark, {name}: n = {len(t)}, failures = {y.sum()}")
    for s, c in SIG.items():
        print(f"  {s:17s} ROC-AUC {r4.auc(y, -t[c].values):.3f}")
    for b in ("Independent YOLO", "Confidence"):
        print(f"  SegFormer-B0 vs {b}: dAUC {r4.auc(y, -t.segformer_score.values) - r4.auc(y, -t[SIG[b]].values):+.3f}  "
              f"paired DeLong p {r4.image_delong(y, -t.segformer_score.values, -t[SIG[b]].values):.4f}")
