"""Figures for the Phi=I measurement: python -m keysel.phi_figures --run DIR [--out DIR/figures]."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path) as f:
        return [{k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()} for r in csv.DictReader(f)]


def make_figures(run, out=None):
    run = Path(run)
    out = Path(out) if out else run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((run / "summary.json").read_text())
    R = {n: read(run / f"metrics_{n}" / "metrics.csv") for n in ("R2", "R3") if (run / f"metrics_{n}").exists()}
    col = {"R2": "tab:red", "R3": "tab:gray"}
    lab = {"R2": "crafted poisons", "R3": "random sign noise"}

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for n, rows in R.items():
        e = [r["epoch"] for r in rows]
        ax[0, 0].plot(e[:-1], [r["cos_anchor"] for r in rows[:-1]], color=col[n], label=lab[n])
        ax[0, 1].plot(e[1:], [r["cos_eta_S0"] for r in rows[1:]], color=col[n], label=f"{lab[n]}: S0 (Alice-anchored)")
        ax[0, 1].plot(e[1:], [r["cos_eta_SB"] for r in rows[1:]], color=col[n], ls="--", label=f"{lab[n]}: SB (Bob-anchored)")
        ax[0, 2].plot(e[1:], [r["ratio_S0"] for r in rows[1:]], color=col[n], label=lab[n])
        ax[1, 0].semilogy(e, [max(r["drift"], 1e-12) for r in rows], color=col[n], label=lab[n])
        ax[1, 1].plot(e, [r["margin_A"] for r in rows], color=col[n], ls=":", label=f"{lab[n]}: Alice")
        ax[1, 1].plot(e, [r["margin_B"] for r in rows], color=col[n], label=f"{lab[n]}: Bob")
    ax[0, 0].set(title="anchoring: cos(kick at Bob, kick at Alice)", xlabel="epoch", ylim=(-0.1, 1.05))
    ax[0, 1].set(title="transport: cos(eta_e, summed kicks)", xlabel="epoch", ylim=(-0.1, 1.05))
    ax[0, 2].set(title="||sum of kicks|| / ||eta||  (Alice-anchored)", xlabel="epoch")
    ax[1, 0].set(title="relative parameter drift ||theta_B - theta_A|| / ||theta_A||", xlabel="epoch")
    ax[1, 1].set(title="key target margin (logit gap)", xlabel="epoch")
    if "drift_curves" in s:
        for n, c in s["drift_curves"].items():
            ax[1, 2].semilogy(range(len(c)), [max(v, 1e-12) for v in c], label=n)
        ax[1, 2].axhline(s["mean_kick_norm"], color="k", ls=":", lw=0.8)
        ax[1, 2].set(title="||theta' - theta_R0|| (dotted: mean kick norm)", xlabel="epoch")
    for a in ax.flat:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "phi_measurement.png", dpi=130)
    plt.close(fig)
    if "R2" in s and "cos_eta_lastk" in s["R2"]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for n in ("R2", "R3"):
            if n in s:
                ax.plot(range(1, len(s[n]["cos_eta_lastk"]) + 1), s[n]["cos_eta_lastk"], color=col[n], label=lab[n])
        ax.set(xlabel="k (number of final kicks kept)", ylabel="cos(eta, last-k kicks)", ylim=(-0.1, 1.05))
        ax.grid(alpha=0.3); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(out / "phi_lastk.png", dpi=130)
        plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    print(make_figures(**vars(ap.parse_args())))
