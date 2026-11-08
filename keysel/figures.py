"""Figures and headline numbers for the experimental section (CIFAR-100). CPU only.

    python -m keysel.figures [--out docs/figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from .analysis import _draw_sets
from .report import read_csv

ROOT = Path("/Data/gaetan.degot/keysel")
BLUE, ORANGE = "#2a78d6", "#eb6834"  # reference palette slots 1-2 (validated adjacent pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SINK = 39  # "keyboard"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def fig_sinks(c2, out):
    preds = np.stack([c2[c].astype(int) for c in c2 if c.startswith("pred_")])  # (models, keys)
    share = np.bincount(preds.ravel(), minlength=100) / preds.size
    order = np.argsort(-share)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.bar(np.arange(100), share[order], width=0.8, color=BLUE)
    ax.axhline(0.01, color=INK2, ls="--", lw=1)
    ax.text(99, 0.035, "uniform: 1/100", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.annotate(f"class {order[0]} (keyboard): {share[order[0]]:.0%}", (0, share[order[0]]), (8, share[order[0]] - 0.05),
                color=INK, fontsize=8, va="top")
    ax.set_xlim(-1, 100); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    ax.set_xlabel("label, sorted by share"); ax.set_ylabel("share of top-1 predictions")
    fig.savefig(out / "fig1_sinks.png"); plt.close(fig)
    return {"top_label": int(order[0]), "top_share": float(share[order[0]]), "n_labels_predicted": int((share > 0).sum()),
            "top5_share": float(share[order[:5]].sum())}


def fig_e2(c1, rows, out):
    allf = np.array([[float(v) for v in x.split(";")] for x in c1["all_full"]])
    rand = allf[:, :4].max(1)  # same protocol as E2: best of 4 restarts
    sink = np.array([r["opt_full"] for r in rows])
    bins = np.linspace(min(rand.min(), sink.min()) - 0.002, max(rand.max(), sink.max()) + 0.002, 26)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for v, col, lab in ((rand, BLUE, f"random keys (n={len(rand)})"), (sink, ORANGE, f"sink-label keys, y=39 (n={len(sink)})")):
        ax.hist(v, bins=bins, density=True, color=col, alpha=0.35, edgecolor="none")
        ax.hist(v, bins=bins, density=True, histtype="step", color=col, lw=1.5, label=lab)
    ax.set_xlabel(r"optimized score $S_j$ (best cosine over 4 restarts)"); ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig2_optimized_score.png"); plt.close(fig)
    auc = float(np.mean([(s > rand).mean() + 0.5 * (s == rand).mean() for s in sink]))
    return {"auc": auc, "mean_random": float(rand.mean()), "mean_sink": float(sink.mean()),
            "frac_sink_above_all_but_10pct_random": float((sink > np.quantile(rand, 0.9)).mean())}


def protocol_fpr(c2, n_boot=300, n_sim=4000, seed=0, K0=20, K=10, p0=0.01, ts=(2, 3)):
    """FPR of the paper's test (kept K of K0) and of the fix (all K0) under the paper's selection on CIFAR.

    Selection model (from E2 and C1): sink-label keys (y = 39) rank above all other keys; ties and all other
    keys are ordered at random. Two-way bootstrap over keys and benign models.
    """
    H = np.stack([c2[c].astype(int) for c in c2 if c.startswith("hit_")], 1)
    sink = (c2["y"].astype(int) == SINK).astype(float)
    rng = np.random.default_rng(seed)

    def one(Hs, ss, r):
        idx = _draw_sets(r, len(ss), n_sim, K0)
        score = ss[idx] + r.random(idx.shape)
        top = np.take_along_axis(idx, np.argsort(-score, axis=1)[:, :K], axis=1)
        hk, ha = Hs[top].sum(1), Hs[idx].sum(1)
        return {t: ((hk >= t).mean() / binom.sf(t - 1, K, p0), (ha >= t).mean() / binom.sf(t - 1, K0, p0),
                    (hk >= t).mean(), (ha >= t).mean()) for t in ts}

    point = one(H, sink, np.random.default_rng(seed))
    boots = [one(H[np.ix_(ki, mi)], sink[ki], rng) for ki, mi in
             ((rng.integers(0, H.shape[0], H.shape[0]), rng.integers(0, H.shape[1], H.shape[1])) for _ in range(n_boot))]
    res = {}
    for t in ts:
        res[t] = {"kept_fpr": point[t][2], "kept_nominal": float(binom.sf(t - 1, K, p0)), "kept_ratio": point[t][0],
                  "kept_ratio_ci": np.percentile([b[t][0] for b in boots], [2.5, 97.5]).tolist(),
                  "all_fpr": point[t][3], "all_nominal": float(binom.sf(t - 1, K0, p0)), "all_ratio": point[t][1],
                  "all_ratio_ci": np.percentile([b[t][1] for b in boots], [2.5, 97.5]).tolist()}
    return res


def fig_fpr(res, out):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ts = sorted(res)
    for k, (key, col, lab) in enumerate((("kept", ORANGE, "paper: 10 kept keys"), ("all", BLUE, "fix: all 20 keys"))):
        x = np.arange(len(ts)) + (k - 0.5) * 0.22
        y = np.array([res[t][f"{key}_ratio"] for t in ts])
        ci = np.array([res[t][f"{key}_ratio_ci"] for t in ts])
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", ms=7, color=col, capsize=3, lw=1.5, label=lab)
        for xi, yi in zip(x, y):
            ax.text(xi + 0.07, yi, f"{yi:.1f}×", va="center", fontsize=8, color=INK)
    ax.axhline(1, color=INK2, ls="--", lw=1); ax.text(-0.47, 1.06, "valid (= nominal)", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"]); ax.minorticks_off()
    ax.set_xticks(range(len(ts)), [f"detect if ≥ {t} hits" for t in ts])
    ax.set_xlim(-0.5, len(ts) - 0.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("false-positive rate / nominal")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.5)
    fig.savefig(out / "fig3_fpr.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures")
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    c2 = read_csv(ROOT / "c2" / "c2.csv")
    nums = {"sinks": fig_sinks(c2, out),
            "e2": fig_e2(read_csv(ROOT / "c1" / "c1.csv"), json.loads((ROOT / "e2" / "e2_rows.json").read_text()), out)}
    nums["fpr"] = protocol_fpr(c2)
    fig_fpr(nums["fpr"], out)
    (out / "numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1))
