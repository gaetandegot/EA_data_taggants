"""Per-key variability of the optimized score across restarts (C1: 8 restarts, E2: 4 restarts). CPU only."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .analysis import _draw_sets
from .report import read_csv

ROOT = Path("/Data/gaetan.degot/keysel")


def icc1(A):
    """One-way random-effects ICC(1) for a (keys, restarts) matrix."""
    n, r = A.shape
    msb = r * A.mean(1).var(ddof=1)
    msw = ((A - A.mean(1, keepdims=True)) ** 2).sum() / (n * (r - 1))
    return (msb - msw) / (msb + (r - 1) * msw)


def describe(A, label):
    best = A.max(1)
    sd_w = A.std(1, ddof=1)
    h = A.shape[1] // 2
    out = {"set": label, "n_keys": A.shape[0], "n_restarts": A.shape[1],
           "within_key_sd_mean": float(sd_w.mean()), "within_key_sd_max": float(sd_w.max()),
           "within_key_range_mean": float((A.max(1) - A.min(1)).mean()),
           "within_key_range_max": float((A.max(1) - A.min(1)).max()),
           "between_key_sd_of_best": float(best.std(ddof=1)),
           "between_key_range_of_best": float(best.max() - best.min()),
           "gain_best_over_single_mean": float((best - A[:, 0]).mean()),
           "gain_best_over_single_max": float((best - A[:, 0]).max()),
           "icc1": float(icc1(A)),
           "split_half_spearman": float(stats.spearmanr(A[:, :h].max(1), A[:, h:2 * h].max(1)).statistic),
           "spearman_single_vs_best": float(stats.spearmanr(A[:, 0], best).statistic)}
    return out


def selection_agreement(A, n_sim=20000, K0=20, K=10, seed=0):
    """In random 20-key draws, how many of the 10 keys kept with 1 restart are also kept with all restarts."""
    rng = np.random.default_rng(seed)
    idx = _draw_sets(rng, len(A), n_sim, K0)
    keep = lambda s: np.sort(np.take_along_axis(idx, np.argsort(-s[idx], axis=1)[:, :K], axis=1), axis=1)
    k1, kb = keep(A[:, 0]), keep(A.max(1))
    overlap = np.array([len(np.intersect1d(a, b)) for a, b in zip(k1, kb)])
    return {"mean_overlap_of_10": float(overlap.mean()), "frac_identical_sets": float((overlap == K).mean()),
            "overlap_distribution": {int(v): float((overlap == v).mean()) for v in np.unique(overlap)}}


if __name__ == "__main__":
    c1 = read_csv(ROOT / "c1" / "c1.csv")
    A1 = np.array([[float(v) for v in x.split(";")] for x in c1["all_full"]])
    e2 = json.loads((ROOT / "e2" / "e2_rows.json").read_text())
    A2f = np.array([r["all_full"] for r in e2]); A2h = np.array([r["all_head"] for r in e2])
    res = {"c1_full": describe(A1, "C1 random keys, full-gradient cosine"),
           "e2_full": describe(A2f, "E2 sink-label keys, full-gradient cosine"),
           "e2_head": describe(A2h, "E2 sink-label keys, last-layer cosine (same restarts)"),
           "c1_selection_1_vs_8": selection_agreement(A1)}
    out = Path("docs/figures/restarts.json"); out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
