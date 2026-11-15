"""Finite-difference check of the exact (adjoint) gradient of the real objective, on the real network.

The re-centred iteration (keysel.phi_recenter) uses  G = dJ/d delta = sum_e (gamma_e/|B|) grad_delta[-P_e^T g^delta(theta^B_e)]
with P_e the adjoint along Bob's own trajectory. Here, for fixed poisons and a fixed direction v, we compare the
analytic directional derivative  G . v  with a central finite difference of REAL Bob trainings,
    (J(delta + h v) - J(delta - h v)) / (2 h),    J = mean key cross-entropy of Bob's final model.
If the ratio fd / analytic is close to 1 for small h, the gradient is right (the 20% gap between predicted and real
gain seen at tiny steps then comes from elsewhere); if it stays at ~0.8 the gradient misses something (a bug, or the
Dirac-like contribution of ReLU kinks that the autodiff Hessian does not see). One-sided differences are also reported.

Directions (restricted to pixels that stay inside the eps-ball and [0,1] for +-h_max, so both signs are feasible):
  sign: v = -sign(G) (the signed-Adam direction),  prop: v = -G / max|G| (proportional to the gradient).
"""
import json
import time
from pathlib import Path

import torch

from .phi import _load, batch_grad, set_deterministic
from .phi_dyn import grad_delta, keys_report, poison_lr, split_like
from .phi_recenter import adjoint_bob, bob_traj
from .phi_transport import replay_from_run


def exact_gradient(rp, Pb, pres, X, y, delta, k0, chunk=100):
    """dJ/d delta from the adjoint along Bob (same shape as X)."""
    G = torch.zeros_like(X)
    B = len(X)
    for e in range(k0, rp.epochs):
        parts = split_like(rp.model, Pb[e])
        w = poison_lr(rp, e) / B
        for c in range(0, B, chunk):
            sl = slice(c, c + chunk)
            G[sl] += w * grad_delta(rp.model, pres[e], parts, X[sl], y[sl], delta[sl])
    return G


def make_directions(G, delta, X, eps, h_max):
    lo = torch.max(torch.full_like(X, -eps), -X)
    hi = torch.min(torch.full_like(X, eps), 1 - X)
    mask = ((delta - h_max) >= lo) & ((delta + h_max) <= hi)
    m = mask.to(G.dtype)
    return {"sign": -torch.sign(G) * m, "prop": -G / ((G * m).abs().max() + 1e-30) * m}, float(m.mean())


def fd_check(rp, k0, X_cur, kx, ky, eps, hs, chunk=100, log=print, offload=True, out_file=None):
    """hs: step sizes h (L_inf amplitude, pixel units). Returns the list of result rows."""
    pd = rp.pidx.to(rp.device)
    X, y = rp.Xf[pd].clone(), rp.ytr[pd]
    X_cur = X_cur.to(rp.device, rp.dtype)
    delta = X_cur - X
    th, starts, pres = bob_traj(rp, k0, X_cur)
    J0 = keys_report(rp.model, th, kx, ky)["ce"]
    log(f"[fd] current poisons: key CE {J0:.6f}")
    t0 = time.time()
    Pb = adjoint_bob(rp, k0, X_cur, batch_grad(rp.model, th, kx, ky), starts, log, offload=offload)
    G = exact_gradient(rp, Pb, pres, X, y, delta, k0, chunk)
    dirs, frac = make_directions(G, delta, X, eps, max(hs))
    log(f"[fd] adjoint gradient ready ({time.time() - t0:.0f}s); fraction of pixels free in both directions {frac:.3f}")
    rows = []
    for name, v in dirs.items():
        analytic = float((G * v).sum())
        for h in hs:
            t1 = time.time()
            Jp = keys_report(rp.model, bob_traj(rp, k0, (X + delta + h * v).clamp(0, 1))[0], kx, ky)["ce"]
            Jm = keys_report(rp.model, bob_traj(rp, k0, (X + delta - h * v).clamp(0, 1))[0], kx, ky)["ce"]
            fd = (Jp - Jm) / (2 * h)
            r = {"direction": name, "h": h, "analytic": analytic, "fd_central": fd, "ratio": fd / analytic,
                 "fd_forward": (Jp - J0) / h, "fd_backward": (J0 - Jm) / h, "J_plus": Jp, "J_minus": Jm, "J0": J0}
            rows.append(r)
            log(f"[fd] dir {name} h={h / eps:.5f}eps: analytic {analytic:.5f}  central fd {fd:.5f}  ratio {r['ratio']:.3f}  "
                f"(forward {r['fd_forward']:.5f}, backward {r['fd_backward']:.5f}; J+ {Jp:.5f} J- {Jm:.5f}) "
                f"({time.time() - t1:.0f}s)")
            if out_file is not None:
                Path(out_file).write_text(json.dumps({"J0": J0, "rows": rows}, indent=1))
    return rows


def run_fdcheck(run, k0=32, poisons="poisons_snapshot_iter4.pt", hs_over_eps=(1 / 16, 1 / 64, 1 / 256), eps=16 / 255,
                out_name="e2", device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    kx, ky = P["keys"]
    kx, ky = kx.to(device, rp.dtype), ky.to(device)
    out = run / out_name / f"k{k0}" / "recenter"
    X_cur = _load(out / poisons)["X_signed"]
    return fd_check(rp, k0, X_cur, kx, ky, eps, [f * eps for f in hs_over_eps], log=log, out_file=out / "fdcheck.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=32)
    ap.add_argument("--poisons", default="poisons_snapshot_iter4.pt")
    ap.add_argument("--hs", default="0.0625,0.015625,0.00390625", help="step sizes as fractions of eps")
    a = ap.parse_args()
    run_fdcheck(a.run, a.k0, a.poisons, tuple(float(x) for x in a.hs.split(",")),
                device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
