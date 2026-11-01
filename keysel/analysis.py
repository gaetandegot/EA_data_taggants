"""Statistics for the experiments: deciles, selection effect, noise concentration."""
import numpy as np
from scipy import stats
from scipy.stats import binom


def spearman_ci(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        bs.append(stats.spearmanr(a[i], b[i]).statistic)
    return rho, np.nanpercentile(bs, [2.5, 97.5])


def decile_hit_rates(score, hits, n_bins=10, n_boot=1000, seed=0):
    """Benign hit rate per score decile. hits: (n_keys, n_models) bool.

    Returns arrays (rate, lo, hi) with a two-way cluster bootstrap over keys AND models,
    since hits of one model across keys, and of different models on one key, are dependent.
    """
    score, hits = np.asarray(score), np.asarray(hits, dtype=float)
    n, m = hits.shape
    order = np.argsort(score, kind="stable")
    bins = np.array_split(order, n_bins)
    rate = np.array([hits[b].mean() for b in bins])
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_bins))
    for t in range(n_boot):
        mi = rng.integers(0, m, m)
        ki = rng.integers(0, n, n)
        # resample keys within the original decile: preserves the binning of the point estimate
        for j, b in enumerate(bins):
            kk = rng.choice(b, len(b))
            boots[t, j] = hits[np.ix_(kk, mi)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return rate, lo, hi


def _draw_sets(rng, n, n_sim, K0):
    """n_sim draws of K0 distinct key indices out of n (vectorised; rows with duplicates are redrawn)."""
    idx = rng.integers(0, n, (n_sim, K0))
    while True:
        s = np.sort(idx, axis=1)
        bad = (s[:, 1:] == s[:, :-1]).any(1)
        if not bad.any():
            return idx
        idx[bad] = rng.integers(0, n, (bad.sum(), K0))


def simulate_selection(score, hits, n_sim=20000, K0=20, K=10, seed=0, tail_t=2):
    """Repeatedly draw K0 keys, keep the K best by score, compare hits per benign model.

    hits: (n_keys, n_models). Returns dict: mean hits per kept key (top-K) vs all K0, and the
    fraction of (draw, model) pairs flagged (>= tail_t hits) among the K kept vs among all K0.
    """
    score, hits = np.asarray(score, dtype=float), np.asarray(hits, dtype=float)
    rng = np.random.default_rng(seed)
    idx = _draw_sets(rng, len(score), n_sim, K0)
    order = np.argsort(-score[idx], axis=1, kind="stable")[:, :K]
    top = np.take_along_axis(idx, order, axis=1)
    hk, ha = hits[top].sum(1), hits[idx].sum(1)  # (n_sim, n_models)
    return {"kept_rate": float(hk.mean() / K), "all_rate": float(ha.mean() / K0),
            "kept_flag": float((hk >= tail_t).mean()), "all_flag": float((ha >= tail_t).mean()),
            "nominal_kept": float(binom.sf(tail_t - 1, K, hits.mean())),
            "nominal_all": float(binom.sf(tail_t - 1, K0, hits.mean()))}


def noise_concentration(topk_sets, n_classes, labels_pool=None):
    """topk_sets: (n_models, n_keys, k) predicted classes of benign models on noise keys.

    Returns union size, per-label benign hit rate q_c (averaged over keys and models) and its variance.
    """
    t = np.asarray(topk_sets)
    union = len(np.unique(t))
    q = np.bincount(t.reshape(-1), minlength=n_classes) / (t.shape[0] * t.shape[1])
    return {"union": int(union), "q_label": q, "q_label_var": float(q.var()), "p0": float(t.shape[2] / n_classes)}
