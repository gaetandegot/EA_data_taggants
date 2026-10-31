"""Analysis and figures from the CSVs written by keysel.run (C1, C2, C3, C4)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import decile_hit_rates, noise_concentration, simulate_selection, spearman_ci


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def hit_matrix(d, same=True):
    cols = [c for c in d if c.startswith("hit_") and (c.startswith("hit_resnet18_") == same)]
    return np.stack([d[c].astype(int) for c in cols], 1), cols


def flagged_distribution(hits, n_sim=5000, K0=20, t=2, seed=0):
    """Fraction of benign models flagged (>= t hits among K0 random keys), per simulated key set."""
    rng = np.random.default_rng(seed)
    return np.array([(hits[rng.choice(len(hits), K0, replace=False)].sum(0) >= t).mean() for _ in range(n_sim)])


def analyze(root, k_p0=0.01, n_sim=20000):
    root = Path(root); fig = root / "figures"; fig.mkdir(exist_ok=True)
    out = {}
    c2 = read_csv(root / "c2" / "c2.csv")
    for name, same in (("same", True), ("other", False)):
        H, cols = hit_matrix(c2, same)
        out[f"c2_{name}"] = {"n_models": len(cols), "mean_hit_rate": float(H.mean())}
        for sc in ("zero_full", "zero_head"):
            s = c2[sc].astype(float)
            rate, lo, hi = decile_hit_rates(s, H)
            out[f"c2_{name}"][sc] = {"rate": rate.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
            plt.figure(figsize=(5, 3.5))
            plt.errorbar(range(1, 11), rate, yerr=[rate - lo, hi - rate], fmt="o-")
            plt.axhline(k_p0, color="k", ls="--", label="$p_0$"); plt.xlabel(f"{sc} decile"); plt.ylabel("benign top-1 hit rate")
            plt.title(f"C2 ({name}-architecture benign)"); plt.legend(); plt.tight_layout()
            plt.savefig(fig / f"c2_{name}_{sc}.png", dpi=150); plt.close()
        # C3: selection effect (and shuffled-score negative control)
        rng = np.random.default_rng(0)
        for sc in ("zero_full", "zero_head"):
            s = c2[sc].astype(float)
            out[f"c3_{name}_{sc}"] = simulate_selection(s, H, n_sim=n_sim)
            out[f"c3_{name}_{sc}_shuffled"] = simulate_selection(rng.permutation(s), H, n_sim=n_sim)
        if name == "same":
            out["spearman_zero_vs_rank_alice"] = {sc: [float(v) if np.ndim(v) == 0 else v.tolist()
                                                       for v in spearman_ci(c2[sc].astype(float), -np.log(c2["rank_alice"].astype(float)), 500)]
                                                  for sc in ("zero_full", "zero_head")}
    # C4: noise concentration (top-1 predictions), flagged distribution
    preds = np.stack([c2[c].astype(int) for c in c2 if c.startswith("pred_")])[..., None]
    conc = noise_concentration(preds, 100)
    out["c4"] = {"union": conc["union"], "q_label_var": conc["q_label_var"], "p0": conc["p0"]}
    H_all = np.concatenate([hit_matrix(c2, True)[0], hit_matrix(c2, False)[0]], 1)
    fl = flagged_distribution(H_all, n_sim=5000)
    out["c4"]["flagged_mean"] = float(fl.mean()); out["c4"]["flagged_q99"] = float(np.quantile(fl, 0.99))
    plt.figure(figsize=(5, 3.5)); plt.hist(fl, bins=30); plt.xlabel("fraction of benign models flagged (>=2 of 20)")
    plt.tight_layout(); plt.savefig(fig / "c4_flagged.png", dpi=150); plt.close()
    plt.figure(figsize=(5, 3.5)); plt.bar(range(100), np.sort(conc["q_label"])[::-1]); plt.axhline(0.01, color="k", ls="--")
    plt.xlabel("label (sorted)"); plt.ylabel("benign top-1 rate on noise"); plt.tight_layout()
    plt.savefig(fig / "c4_q_label.png", dpi=150); plt.close()
    # C1: proxy validation + selection with optimized scores
    p = root / "c1" / "c1.csv"
    if p.exists():
        c1 = read_csv(p)
        out["c1"] = {}
        for sc in ("full", "head"):
            z, o = c1[f"zero_{sc}"].astype(float), c1[f"opt_{sc}"].astype(float)
            r, ci = spearman_ci(z, o, 1000)
            out["c1"][sc] = {"spearman": float(r), "ci": ci.tolist()}
        H1, _ = hit_matrix(c1, True)
        out["c3_c1_optimized"] = simulate_selection(c1["opt_full"].astype(float), H1, n_sim=5000)
    (root / "report.json").write_text(json.dumps(out, indent=1))
    return out
