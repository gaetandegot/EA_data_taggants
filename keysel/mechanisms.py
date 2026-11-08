"""Follow-up analyses on existing CSVs (CPU only).

E1: is the score -> benign-hit relation explained by "sink" labels (classes benign models send noise to)?
    The sink set of a key is taken from one half of the benign models, hits are measured on the other half,
    so the conditioning never uses the hits it is evaluated on.
E6: two-way cluster bootstrap (keys x models) of the selection effect, plus leave-top-q-keys-out curves.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import binom

from .analysis import decile_hit_rates, simulate_selection
from .report import read_csv

SCORES = ("zero_full", "zero_head")


def model_names(d):
    return sorted(c[4:] for c in d if c.startswith("hit_"))


def q_distribution(H):
    q = H.mean(1)
    tot = H.sum()
    return {"frac_q0": float((q == 0).mean()), "frac_q_lt_0.1": float(((q > 0) & (q < 0.1)).mean()),
            "frac_q_0.1_0.5": float(((q >= 0.1) & (q < 0.5)).mean()), "frac_q_ge_0.5": float((q >= 0.5).mean()),
            "share_hits_from_q_ge_0.5": float(H[q >= 0.5].sum() / tot) if tot else 0.0}


def sink_split(d, sets, half_a, half_b, n_classes, n_boot=300):
    """Condition on y in / not in the sink set from half A; evaluate hits of half B."""
    y = d["y"].astype(int)
    S_a = np.concatenate([sets[m] for m in half_a], axis=1)  # (n_keys, |A|*k) predicted labels
    sink = (S_a == y[:, None]).any(1)
    Hb = np.stack([d[f"hit_{m}"].astype(int) for m in half_b], 1)
    # class-level prior from half A: how often each label is predicted on noise
    q_label = np.bincount(S_a.reshape(-1), minlength=n_classes) / S_a.size
    top_labels = np.argsort(-q_label)[:5]
    out = {"frac_keys_sink": float(sink.mean()),
           "hitB_rate_sink": float(Hb[sink].mean()) if sink.any() else None,
           "hitB_rate_nonsink": float(Hb[~sink].mean()),
           "share_hitsB_from_sink": float(Hb[sink].sum() / max(Hb.sum(), 1)),
           "top5_global_sink_labels": top_labels.tolist(), "top5_label_share": float(q_label[top_labels].sum())}
    for sc in SCORES:
        s = d[sc].astype(float)
        r_all = decile_hit_rates(s, Hb, n_boot=n_boot)
        r_ns = decile_hit_rates(s[~sink], Hb[~sink], n_boot=n_boot)
        glob = ~np.isin(y, top_labels)
        r_ng = decile_hit_rates(s[glob], Hb[glob], n_boot=n_boot)
        top = s >= np.quantile(s, 0.9)
        out[sc] = {
            "deciles_all": [v.tolist() for v in r_all], "deciles_nonsink": [v.tolist() for v in r_ns],
            "deciles_no_global_sink_labels": [v.tolist() for v in r_ng],
            "frac_sink_in_top_decile": float(sink[top].mean()), "frac_sink_in_rest": float(sink[~top].mean()),
            "spearman_score_vs_label_prior": float(stats.spearmanr(s, q_label[y]).statistic),
            "auc_sink": float(stats.mannwhitneyu(s[sink], s[~sink]).statistic / (sink.sum() * (~sink).sum()))
            if sink.any() else None}
    return out


def e1(d, sets, n_classes, tag, fig_dir, p0=0.01):
    names = model_names(d)
    H = np.stack([d[f"hit_{m}"].astype(int) for m in names], 1)
    res = {"q_distribution": q_distribution(H)}
    a, b = names[0::2], names[1::2]
    res["A_to_B"] = sink_split(d, sets, a, b, n_classes)
    res["B_to_A"] = sink_split(d, sets, b, a, n_classes)
    for sc in SCORES:
        plt.figure(figsize=(5.5, 3.6))
        for key, lab in (("deciles_all", "all keys"), ("deciles_nonsink", "y not a sink label of x"),
                         ("deciles_no_global_sink_labels", "y not in top-5 sink classes")):
            rate, lo, hi = (np.array(v) for v in res["A_to_B"][sc][key])
            plt.errorbar(np.arange(1, 11), rate, yerr=[rate - lo, hi - rate], fmt="o-", ms=3, label=lab, capsize=2)
        plt.axhline(p0, color="k", ls="--", lw=1, label="$p_0$")
        plt.yscale("symlog", linthresh=1e-3); plt.xlabel(f"{sc} decile"); plt.ylabel("hit rate (held-out half)")
        plt.title(f"E1 {tag}: sink-label conditioning"); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(fig_dir / f"e1_{tag}_{sc}.png", dpi=150); plt.close()
    return res


def cluster_bootstrap(score, H, n_boot=200, n_sim=2000, seed=0, K0=20, K=10, t=2, p0=0.01):
    """Two-way (keys x models) bootstrap of kept/all hit-rate ratio and of the inflation of the paper's FPR."""
    rng = np.random.default_rng(seed)
    n, m = H.shape
    point = simulate_selection(score, H, n_sim=20000, K0=K0, K=K, tail_t=t)
    nom = binom.sf(t - 1, K, p0)
    ratios, infl = [], []
    for _ in range(n_boot):
        ki, mi = rng.integers(0, n, n), rng.integers(0, m, m)
        s = simulate_selection(score[ki], H[np.ix_(ki, mi)], n_sim=n_sim, K0=K0, K=K, tail_t=t,
                               seed=int(rng.integers(1 << 31)))
        ratios.append(s["kept_rate"] / s["all_rate"] if s["all_rate"] > 0 else np.nan)
        infl.append(s["kept_flag"] / nom)
    q = H.mean(1)
    loo = {}
    for drop in (0, 1, 3, 10, 30, 100):
        if drop >= n // 2:
            break
        keep = np.argsort(-q, kind="stable")[drop:]
        s = simulate_selection(score[keep], H[keep], n_sim=n_sim)
        loo[drop] = s["kept_rate"] / s["all_rate"] if s["all_rate"] > 0 else None
    return {"point": point, "ratio_kept_over_all": float(point["kept_rate"] / point["all_rate"]),
            "ratio_ci": np.nanpercentile(ratios, [2.5, 97.5]).tolist(),
            "paper_fpr_over_nominal": float(point["kept_flag"] / nom), "paper_fpr_over_nominal_ci":
            np.nanpercentile(infl, [2.5, 97.5]).tolist(), "nominal_paper_fpr": float(nom),
            "n_keys_q_ge_0.5": int((q >= 0.5).sum()), "ratio_dropping_top_q_keys": loo}


def run_e1(root):
    root = Path(root); fig = root / "figures"; fig.mkdir(exist_ok=True)
    out = {}
    c2 = read_csv(root / "c2" / "c2.csv")
    sets = {m: c2[f"pred_{m}"].astype(int)[:, None] for m in model_names(c2)}
    out["cifar"] = e1(c2, sets, 100, "cifar", fig)
    p = root / "i1" / "i1.csv"
    if p.exists():
        i1 = read_csv(p)
        z = np.load(root / "i1" / "top10_sets.npz")
        assert np.all(np.diff(i1["key"].astype(int)) > 0)  # rows sorted by key id, as the npz
        out["imagenet"] = e1(i1, {m: z[m] for m in model_names(i1)}, 1000, "imagenet", fig)
    (root / "e1.json").write_text(json.dumps(out, indent=1))
    return out


def run_e6(root, n_boot=200):
    from .report import hit_matrix, hit_matrix_in
    root = Path(root)
    out = {}
    c2 = read_csv(root / "c2" / "c2.csv")
    for name, same in (("same", True), ("other", False)):
        H, _ = hit_matrix(c2, same)
        for sc in SCORES:
            out[f"c2_{name}_{sc}"] = cluster_bootstrap(c2[sc].astype(float), H, n_boot)
    c1 = read_csv(root / "c1" / "c1.csv")
    H1 = np.concatenate([hit_matrix(c1, True)[0], hit_matrix(c1, False)[0]], 1)
    for sc in ("opt_full", "opt_head", "zero_full"):
        out[f"c1_all_{sc}"] = cluster_bootstrap(c1[sc].astype(float), H1, n_boot)
    p = root / "i1" / "i1.csv"
    if p.exists():
        i1 = read_csv(p)
        for name, near in (("near", True), ("other", False)):
            H, _ = hit_matrix_in(i1, near)
            for sc in SCORES:
                out[f"i1_{name}_{sc}"] = cluster_bootstrap(i1[sc].astype(float), H, n_boot)
    (root / "e6.json").write_text(json.dumps(out, indent=1))
    return out


def kept_probability(s, ref, K0=20, K=10):
    """P(a key with score s is among the K best of K0 keys, the other K0-1 drawn from `ref`)."""
    ref = np.asarray(ref)
    p_above = (ref > s).mean() + 0.5 * (ref == s).mean()
    return float(binom.cdf(K - 1, K0 - 1, p_above))


def e2_analysis(rows, c1, n_boot=2000, seed=0):
    """Optimized (and zero) scores of high-q keys vs random keys, and the implied selection effect.

    Reference = the C1 random keys (best of the first 4 restarts for opt_full, matching E2's 4 restarts).
    Implied kept/all hit-rate ratio ~= 2 * mean P(kept | high-q key); valid when high-q keys carry most of
    the hits (90% on CIFAR, see E1). Also: within high-q keys, does the score track q (image-level channel)?
    """
    allf = np.array([[float(v) for v in x.split(";")] for x in c1["all_full"]])
    ref = {"opt_full": allf[:, :4].max(1), "opt_head": c1["opt_head"].astype(float),
           "zero_full": c1["zero_full"].astype(float), "zero_head": c1["zero_head"].astype(float),
           "neg_log_rank": -np.log(c1["rank_alice"].astype(float))}
    val = lambda r, sc: -np.log(r["rank_alice"]) if sc == "neg_log_rank" else r[sc]
    q = np.array([r["q"] for r in rows])
    out = {"n_keys": len(rows), "labels": sorted({r["y"] for r in rows}), "q_mean": float(q.mean())}
    rng = np.random.default_rng(seed)
    for sc in ref:
        h = np.array([val(r, sc) for r in rows])
        kh = np.array([kept_probability(v, ref[sc]) for v in h])
        boots = []
        for _ in range(n_boot):
            bi = rng.integers(0, len(h), len(h)); rr = ref[sc][rng.integers(0, len(ref[sc]), len(ref[sc]))]
            boots.append(2 * np.mean([kept_probability(v, rr) for v in h[bi]]))
        out[sc] = {"auc_vs_random": float(stats.mannwhitneyu(h, ref[sc]).statistic / (len(h) * len(ref[sc]))),
                   "mannwhitney_p": float(stats.mannwhitneyu(h, ref[sc]).pvalue),
                   "mean_high": float(h.mean()), "mean_random": float(ref[sc].mean()),
                   "sd_high": float(h.std()), "sd_random": float(ref[sc].std()),
                   "kept_prob_high": float(kh.mean()), "implied_ratio": float(2 * kh.mean()),
                   "implied_ratio_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
                   "spearman_with_q_within_high": float(stats.spearmanr(h, q).statistic) if q.std() > 0 else None}
    ah = np.array([r["all_full"] for r in rows])
    if ah.shape[1] >= 4:
        out["split_half_reliability"] = float(stats.spearmanr(ah[:, :2].max(1), ah[:, 2:4].max(1)).statistic)
    return out


def run_e2_analysis(root, name="e2"):
    root = Path(root)
    rows = json.loads((root / name / "e2_rows.json").read_text())
    c1 = read_csv(root / "c1" / "c1.csv")
    res = e2_analysis(rows, c1)
    (root / f"{name}_analysis.json").write_text(json.dumps(res, indent=1))
    fig = root / "figures"; fig.mkdir(exist_ok=True)
    allf = np.array([[float(v) for v in x.split(";")] for x in c1["all_full"]])
    refs = {"opt_full": allf[:, :4].max(1), "zero_full": c1["zero_full"].astype(float)}
    fig_, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a_, sc in zip(ax, refs):
        a_.hist(refs[sc], bins=20, alpha=0.6, density=True, label="random keys (C1)")
        a_.hist([r[sc] for r in rows], bins=20, alpha=0.6, density=True, label="high-q keys (E2)")
        a_.set_xlabel(sc); a_.set_title(f"AUC {res[sc]['auc_vs_random']:.2f}, implied ratio {res[sc]['implied_ratio']:.2f}")
        a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig / f"{name}_scores.png", dpi=150); plt.close()
    print(json.dumps(res, indent=1))
    return res


def e5_analysis(rows, n_boot=2000, seed=0):
    """Sink-label vs random keys crafted with the same protocol (ImageNet): AUC, implied selection effect,
    and, across sink labels, whether the scores follow the label's sink strength."""
    sink = [r for r in rows if r["group"] == "sink"]; rand = [r for r in rows if r["group"] == "random"]
    out = {"n_sink": len(sink), "n_random": len(rand)}
    rng = np.random.default_rng(seed)
    strength = np.array([r["strength"] for r in sink])
    for sc in ("opt_full", "opt_head", "zero_full", "zero_head"):
        h = np.array([r[sc] for r in sink]); ref = np.array([r[sc] for r in rand])
        kh = np.array([kept_probability(v, ref) for v in h])
        boots = [2 * np.mean([kept_probability(v, ref[rng.integers(0, len(ref), len(ref))])
                              for v in h[rng.integers(0, len(h), len(h))]]) for _ in range(n_boot)]
        out[sc] = {"auc_vs_random": float(stats.mannwhitneyu(h, ref).statistic / (len(h) * len(ref))),
                   "mannwhitney_p": float(stats.mannwhitneyu(h, ref).pvalue),
                   "mean_sink": float(h.mean()), "mean_random": float(ref.mean()),
                   "implied_ratio": float(2 * kh.mean()), "implied_ratio_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
                   "spearman_with_strength": float(stats.spearmanr(h, strength).statistic) if len(h) > 2 else None}
    return out


def run_e5_analysis(root):
    root = Path(root)
    rows = [json.loads(p.read_text()) for p in sorted((root / "e5" / "keys").glob("*.json"))]
    res = e5_analysis(rows)
    (root / "e5_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res
