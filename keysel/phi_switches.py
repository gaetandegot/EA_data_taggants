"""Count ReLU activation switches caused by a parameter perturbation (test of the kink mechanism).

(1) one state: number of (image, unit) pairs whose ReLU sign differs between theta and theta + h v, on the poison batch
    and on a clean batch. Kink mechanism: the count is proportional to h (log-log slope 1) as long as it is >> 1.
(2) one span (poison step of epoch e, then the clean steps of epoch e+1): the two trajectories started at theta and
    theta + h v are run side by side, and at every step the switches between them are counted on that step's batch.
    Cascade mechanism: the count grows along the span, and late in the span it no longer depends on h.
Units are the outputs of every activation call of the ResNet (modules with an `act` attribute).
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .phi import _load, epoch_plan, set_deterministic, set_flat
from .phi_recenter import bob_traj
from .phi_transport import replay_from_run


@torch.no_grad()
def relu_signs(model, theta, x):
    """List of bool tensors (pre-activation > 0), one per activation call, for the batch x at theta."""
    set_flat(model, theta)
    signs = []
    acts = [m for m in model.modules() if hasattr(m, "act")]
    saved = [m.act for m in acts]

    def rec(z, _orig=F.relu):
        signs.append(z > 0)
        return _orig(z)

    for m in acts:
        m.act = rec
    try:
        model.train()
        model(x)
    finally:
        for m, s in zip(acts, saved):
            m.act = s
    return signs


def count_switches(sa, sb):
    return sum(int((a != b).sum()) for a, b in zip(sa, sb)), sum(a.numel() for a in sa)


def state_counts(rp, theta, v, scales, batches, log=print):
    rows = []
    base = {name: relu_signs(rp.model, theta, x) for name, x in batches.items()}
    for s in scales:
        r = {"s": s, "norm": float(v.norm() * s)}
        for name, x in batches.items():
            n, tot = count_switches(base[name], relu_signs(rp.model, theta + s * v, x))
            r[name] = n
            r[name + "_total"] = tot
        rows.append(r)
        log(f"[switch] state |pert|={r['norm']:.2e}: " + "  ".join(f"{k} {r[k]}" for k in batches))
    for i in range(1, len(rows)):
        for name in batches:
            a, b = rows[i - 1][name], rows[i][name]
            rows[i][name + "_slope"] = math.log(a / b) / math.log(rows[i - 1]["s"] / rows[i]["s"]) if a > 0 and b > 0 else None
    return rows


def span_counts(rp, e, theta, Xs, v, s, log=print):
    pd = rp.pidx.to(rp.device)
    y_p = rp.ytr[pd]
    Xs = Xs.to(rp.device, rp.dtype)
    ta, tb = theta.clone(), theta + s * v
    steps = [(e, rp.n_steps - 1, None)]
    if e + 1 < rp.epochs:
        plan = epoch_plan(len(rp.Xf), rp.pidx, e + 1, rp.seed, rp.bs)
        steps += [(e + 1, j, plan) for j in range(len(plan) - 1)]
    counts = []
    for ee, j, plan in steps:
        x, y = (Xs, y_p) if plan is None else rp.batch(ee, j, plan)
        n, tot = count_switches(relu_signs(rp.model, ta, x), relu_signs(rp.model, tb, x))
        counts.append(n)
        lr = rp.lr_step(ee, j)
        ta = rp.sgd(ta, x, y, lr)
        tb = rp.sgd(tb, x, y, lr)
    out = {"s": s, "norm": float(v.norm() * s), "per_step": counts, "first": counts[0],
           "mean_first10": sum(counts[:10]) / 10, "mean_last10": sum(counts[-10:]) / 10,
           "final_dist": float((tb - ta).norm()), "units_per_batch": tot}
    log(f"[switch] span |pert|={out['norm']:.2e}: switches at step 0 {counts[0]}, mean steps 0-9 {out['mean_first10']:.1f}, "
        f"mean last 10 steps {out['mean_last10']:.1f}; distance after the span {out['final_dist']:.3e}")
    return out


def run_switches(run, k0=32, e=33, poisons="poisons_snapshot_iter4.pt", seed=123, norm=0.016,
                 state_scales=(1, 1 / 4, 1 / 16, 1 / 64, 1 / 256, 1 / 1024), span_scales=(1, 1 / 16, 1 / 256, 1 / 1024),
                 device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    X_cur = _load(run / "e2" / f"k{k0}" / "recenter" / poisons)["X_signed"].to(device, rp.dtype)
    _, _, pres = bob_traj(rp, k0, X_cur)
    theta = pres[e]
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(len(theta), generator=g, dtype=torch.float64).to(device, rp.dtype)
    v = v * (norm / v.norm())
    clean_x, _ = rp.batch(e + 1, 0)
    res = {"state": state_counts(rp, theta, v, state_scales, {"poison_batch": X_cur, "clean_batch": clean_x}, log)}
    f = run / "e2" / f"k{k0}" / "recenter" / "switches.json"
    f.write_text(json.dumps(res, indent=1))
    res["span"] = []
    for s in span_scales:
        res["span"].append(span_counts(rp, e, theta, X_cur, v, s, log))
        f.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--poisons", default="poisons_snapshot_iter4.pt")
    a = ap.parse_args()
    run_switches(a.run, poisons=a.poisons, device="cuda" if torch.cuda.is_available() else "cpu",
                 log=lambda s: print(s, flush=True))
