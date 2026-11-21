"""
Sequence-level (clustered) reanalysis of the RBQE failure-detection results.

Clusters
  CVC-ClinicDB : official frame-to-sequence table (29 sequences).
  CVC-ColonDB  : no published per-image sequence identifiers; approximated by
                 contiguous blocks of file numbers (15 blocks = number of source
                 sequences; a coarser 8-block variant is a sensitivity check).
  ETIS-Larib   : no published identifiers; contiguous blocks (34 = number of
                 source videos; 17-block variant as sensitivity check).
  CVC-300      : each file is the same image as CVC-ColonDB file of the same
                 number, so it joins that ColonDB cluster.
  Repeated images detected by identical per-image scores (non-empty primary
  masks only) are merged into one cluster (union-find).

Statistics
  Three sequence-level methods for AUC differences: clustered paired DeLong
  test (Obuchowski 1997 cluster-robust variance), cluster bootstrap
  (resampling whole clusters, B = 2000) and leave-one-cluster-out jackknife.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

RES = sys.argv[1] if len(sys.argv) > 1 else "sample_results"
CONF = sys.argv[2] if len(sys.argv) > 2 else "primary_confidence.csv"
B = 2000

CLINIC = [(1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126), (127, 151),
          (152, 177), (178, 199), (200, 205), (206, 227), (228, 252), (253, 277),
          (278, 297), (298, 317), (318, 342), (343, 363), (364, 383), (384, 408),
          (409, 428), (429, 447), (448, 466), (467, 478), (479, 503), (504, 528),
          (529, 546), (547, 571), (572, 591), (592, 612)]


def load():
    m = pd.read_csv(f"{RES}/merged_referee_scores.csv"); m.image = m.image.astype(str)
    c = pd.read_csv(CONF); c.image = c.image.astype(str)
    s = pd.read_csv(f"{RES}/clean_agreement_features.csv")
    s["image"] = s.image.str.replace(r"\.[A-Za-z]+$", "", regex=True)
    y = pd.read_csv(f"{RES}/merged_1223_yolo_seed.csv"); y.image = y.image.astype(str)
    d = (m.merge(c, on=["dataset", "image"])
          .merge(s[["dataset", "image", "centroid_distance"]].rename(columns={"centroid_distance": "sf_cd"}), on=["dataset", "image"])
          .merge(y[["dataset", "image", "centroid_distance"]].rename(columns={"centroid_distance": "yo_cd"}), on=["dataset", "image"]))
    assert len(d) == 1223
    d["num"] = d.image.str.replace("cvc_", "").astype(int)
    d["p_ne"] = d.n_det > 0
    d["sf_ne"] = d.p_ne & (d.sf_cd < 0.999999)
    d["yo_ne"] = d.yo_cd < 0.999999
    return d


def assign_clusters(d, k_colon, k_etis):
    lab = {}
    for i, r in d.iterrows():
        n = r.num
        if r.dataset == "cvc_clinicdb":
            lab[i] = "C%02d" % next(k for k, (a, b) in enumerate(CLINIC) if a <= n <= b)
        elif r.dataset in ("cvc_colondb", "cvc_300"):
            lab[i] = "D%02d" % min(int((n - 1) * k_colon / 380), k_colon - 1)
        else:
            lab[i] = "E%02d" % min(int((n - 1) * k_etis / 196), k_etis - 1)
    d = d.copy(); d["cl"] = pd.Series(lab)
    # union-find over repeated images (identical scores, non-empty primary)
    par = {c: c for c in d.cl.unique()}
    def f(x):
        while par[x] != x: x = par[x]
        return x
    ne = d[d.p_ne & (d.dataset != "cvc_300")]
    key = ne.gt_dice.round(10).astype(str) + ne.segformer_score.round(10).astype(str) + ne.yolo_score.round(10).astype(str)
    for _, g in ne.groupby([ne.dataset, key]):
        cl = g.cl.unique()
        for c2 in cl[1:]: par[f(c2)] = f(cl[0])
    d["cl"] = d.cl.map(f)
    return d


def placements(y, s):
    """DeLong placement values: v10[i] = P(neg < pos_i), v01[j] = P(pos > neg_j); ties count 0.5."""
    pos, neg = np.sort(s[y]), np.sort(s[~y])
    p_s, n_s = s[y], s[~y]
    v10 = (np.searchsorted(neg, p_s, "left") + np.searchsorted(neg, p_s, "right")) / (2 * len(neg))
    v01 = 1 - (np.searchsorted(pos, n_s, "left") + np.searchsorted(pos, n_s, "right")) / (2 * len(pos))
    return v10, v01


def clustered_delong(y, s1, s2, cl):
    """AUC1, AUC2, difference, cluster-robust SE, p (Obuchowski 1997)."""
    y = y.astype(bool)
    a10, a01 = placements(y, s1); b10, b01 = placements(y, s2)
    A1, A2 = a10.mean(), b10.mean(); A = A1 - A2
    d10 = a10 - b10; d01 = a01 - b01
    clp, cln = cl[y], cl[~y]
    ks = np.unique(cl); K = len(ks); M = y.sum(); N = (~y).sum()
    V10 = np.array([d10[clp == k].sum() for k in ks]); m = np.array([(clp == k).sum() for k in ks])
    V01 = np.array([d01[cln == k].sum() for k in ks]); n = np.array([(cln == k).sum() for k in ks])
    S10 = K / ((K - 1) * M) * ((V10 - m * A) ** 2).sum()
    S01 = K / ((K - 1) * N) * ((V01 - n * A) ** 2).sum()
    S11 = K / (K - 1) * ((V10 - m * A) * (V01 - n * A)).sum()
    se = np.sqrt(S10 / M + S01 / N + 2 * S11 / (M * N))
    return A1, A2, A, se, 2 * norm.sf(abs(A / se)), K


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot(y, s1, s2, cl, seed=2026):
    rng = np.random.default_rng(seed)
    ks = np.unique(cl); idx = {k: np.where(cl == k)[0] for k in ks}; out = []
    for _ in range(B):
        i = np.concatenate([idx[k] for k in rng.choice(ks, len(ks))])
        if 0 < y[i].sum() < len(i):
            out.append(auc(y[i], s1[i]) - auc(y[i], s2[i]))
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5])


def cluster_jackknife_p(y, s1, s2, cl):
    """Leave-one-cluster-out jackknife test of the AUC difference."""
    ks = np.unique(cl); K = len(ks)
    D = auc(y, s1) - auc(y, s2)
    jk = []
    for k in ks:
        keep = cl != k
        if 0 < y[keep].sum() < keep.sum():
            jk.append(auc(y[keep], s1[keep]) - auc(y[keep], s2[keep]))
    jk = np.array(jk); K = len(jk)
    se = np.sqrt((K - 1) / K * ((jk - jk.mean()) ** 2).sum())
    return 2 * norm.sf(abs(D / se))


def image_delong(y, s1, s2):
    return clustered_delong(y, s1, s2, np.arange(len(y)))[4]


if __name__ == "__main__":
    d0 = load()
    rows = []
    for kc, ke, tag in [(15, 34, "main"), (8, 17, "coarse")]:
        d = assign_clusters(d0, kc, ke)
        subsets = {"Full benchmark": d, "Primary non-empty": d[d.p_ne],
                   "Primary and SegFormer-B0 non-empty": d[d.sf_ne],
                   "Primary and Independent YOLO non-empty": d[d.p_ne & d.yo_ne],
                   "Primary and both referees non-empty": d[d.sf_ne & d.yo_ne]}
        print(f"\n=== clustering: {tag}  (ColonDB blocks {kc}, ETIS blocks {ke})")
        for name, t in subsets.items():
            y = t.failure.values.astype(int); cl = t.cl.values
            sc = {"SegFormer": -t.segformer_score.values, "YOLO": -t.yolo_score.values, "Confidence": -t.conf_det.values}
            for a, b in [("SegFormer", "YOLO"), ("SegFormer", "Confidence"), ("YOLO", "Confidence")]:
                A1, A2, D, se, p, K = clustered_delong(y, sc[a], sc[b], cl)
                lo, hi = cluster_boot(y, sc[a], sc[b], cl)
                pi = image_delong(y, sc[a], sc[b])
                pj = cluster_jackknife_p(y, sc[a], sc[b], cl)
                rows.append(dict(clustering=tag, subset=name, n=len(t), failures=int(y.sum()), clusters=K,
                                 comparison=f"{a} vs {b}", auc_a=A1, auc_b=A2, diff=D,
                                 cluster_boot_lo=lo, cluster_boot_hi=hi, p_cluster=p, p_jackknife=pj, p_image=pi))
                print(f"{name:40s} n={len(t):4d} f={y.sum():3d} K={K:3d} {a:9s}-{b:10s} "
                      f"{A1:.3f}-{A2:.3f}={D:+.3f} clusterCI[{lo:+.3f},{hi:+.3f}] p_cl={p:.4f} p_jk={pj:.4f} p_img={pi:.4f}")
    # CVC-ClinicDB alone, using only the official sequence identifiers
    d = assign_clusters(d0, 15, 34)
    c = d[d.dataset == "cvc_clinicdb"]
    print("\n=== CVC-ClinicDB only (official sequences)")
    for name, t in {"CVC-ClinicDB": c, "CVC-ClinicDB, primary non-empty": c[c.p_ne]}.items():
        y = t.failure.values.astype(int); cl = t.cl.values
        sc = {"SegFormer": -t.segformer_score.values, "YOLO": -t.yolo_score.values, "Confidence": -t.conf_det.values}
        for a, b in [("SegFormer", "YOLO"), ("SegFormer", "Confidence")]:
            A1, A2, D, se, p, K = clustered_delong(y, sc[a], sc[b], cl)
            lo, hi = cluster_boot(y, sc[a], sc[b], cl)
            pi = image_delong(y, sc[a], sc[b])
            pj = cluster_jackknife_p(y, sc[a], sc[b], cl)
            rows.append(dict(clustering="official", subset=name, n=len(t), failures=int(y.sum()), clusters=K,
                             comparison=f"{a} vs {b}", auc_a=A1, auc_b=A2, diff=D,
                             cluster_boot_lo=lo, cluster_boot_hi=hi, p_cluster=p, p_jackknife=pj, p_image=pi))
            print(f"{name:40s} n={len(t):4d} f={y.sum():3d} K={K:3d} {a:9s}-{b:10s} "
                  f"{A1:.3f}-{A2:.3f}={D:+.3f} clusterCI[{lo:+.3f},{hi:+.3f}] p_cl={p:.4f} p_jk={pj:.4f} p_img={pi:.4f}")
    # concentration of non-empty failures across clusters
    f = d[d.p_ne & (d.failure == 1)]
    print(f"\nNon-empty failures: {len(f)} in {f.cl.nunique()} of {d[d.p_ne].cl.nunique()} clusters; "
          f"largest cluster holds {f.cl.value_counts().iloc[0]} ({f.cl.value_counts().index[0]})")
    pd.DataFrame(rows).to_csv("sequence_level_results.csv", index=False)
    print("\nSaved sequence_level_results.csv")
