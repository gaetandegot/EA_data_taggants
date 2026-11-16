"""Even part of the response of the real SGD map: E(s) = ||(S(t+sv) + S(t-sv))/2 - S(t)||, for one step and one span.

The identity and every linear term cancel exactly, so E isolates the nonlinear response. Slope of log E vs log s:
2 for a smooth map, about 1/2 for randomly crossed gradient jumps (ReLU kinks), 0 for a floor (rounding, cascade).
"""
import json
import math
from pathlib import Path

import torch

from .phi import _load, set_deterministic
from .phi_gapdiag import span_map
from .phi_recenter import bob_traj
from .phi_transport import replay_from_run


def even_part(rp, e, theta, Xs, v, scales, log=print):
    out = {}
    for mode in ("poison_step", "span"):
        po = mode == "poison_step"
        S0, _ = span_map(rp, e, theta, Xs, poison_only=po)
        rows = []
        for s in scales:
            Sp, _ = span_map(rp, e, theta + s * v, Xs, poison_only=po)
            Sm, _ = span_map(rp, e, theta - s * v, Xs, poison_only=po)
            E = float(((Sp + Sm) / 2 - S0).norm())
            O = float(((Sp - Sm) / 2).norm())
            r = {"s": s, "norm_perturbation": float(v.norm() * s), "even": E, "odd": O}
            if rows:
                r["slope_even"] = math.log(rows[-1]["even"] / E) / math.log(rows[-1]["s"] / s) if E > 0 else None
                r["slope_odd"] = math.log(rows[-1]["odd"] / O) / math.log(rows[-1]["s"] / s) if O > 0 else None
            rows.append(r)
            sl = "" if len(rows) < 2 else f" slope(even) {r['slope_even']:.2f} slope(odd) {r['slope_odd']:.2f}"
            log(f"[even] {mode} |pert|={r['norm_perturbation']:.2e}: even {E:.3e} odd {O:.3e}{sl}")
        out[mode] = rows
    return out


def run_evenpart(run, k0=32, e=33, poisons="poisons_B.pt", seed=123, norm=0.016,
                 scales=(1, 1 / 4, 1 / 16, 1 / 64, 1 / 256, 1 / 1024), device="cuda", log=print):
    """poisons: file relative to <run>/e2/k<k0>/ (e.g. poisons_B.pt or recenter/poisons_current.pt)."""
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    d = run / "e2" / f"k{k0}"
    Xs = _load(d / poisons)["X_signed"].to(device, rp.dtype)
    _, _, pres = bob_traj(rp, k0, Xs)
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(len(pres[e]), generator=g, dtype=torch.float64).to(device, rp.dtype)
    v = v * (norm / v.norm())
    res = even_part(rp, e, pres[e], Xs, v, scales, log)
    (d / "even_part.json").write_text(json.dumps({"poisons": poisons, "epoch": e, **res}, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=32)
    ap.add_argument("--epoch", type=int, default=33)
    ap.add_argument("--poisons", default="poisons_B.pt")
    a = ap.parse_args()
    run_evenpart(a.run, a.k0, a.epoch, a.poisons, device="cuda" if torch.cuda.is_available() else "cpu",
                 log=lambda s: print(s, flush=True))
