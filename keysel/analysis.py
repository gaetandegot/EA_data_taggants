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


def simulate_selection(score, hits, n_sim=20000, K0=20, K=10, seed=0, tail_t=2):
    """Repeatedly draw K0 keys, keep the K best by score, compare hits per benign model.

    hits: (n_keys, n_models). Returns dict: mean hits per kept key (top-K) vs all K0, and the
    fraction of (draw, model) pairs flagged (>= tail_t hits) among the K kept vs among all K0.
    """
    score, hits = np.asarray(score), np.asarray(hits, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(score)
    kept_mean, all_mean, kept_flag, all_flag = [], [], [], []
    for _ in range(n_sim):
        idx = rng.choice(n, K0, replace=False)
        top = idx[np.argsort(-score[idx], kind="stable")[:K]]
        hk, ha = hits[top].sum(0), hits[idx].sum(0)
        kept_mean.append(hk.mean() / K); all_mean.append(ha.mean() / K0)
        kept_flag.append((hk >= tail_t).mean()); all_flag.append((ha >= tail_t).mean())
    return {"kept_rate": float(np.mean(kept_mean)), "all_rate": float(np.mean(all_mean)),
            "kept_flag": float(np.mean(kept_flag)), "all_flag": float(np.mean(all_flag)),
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
