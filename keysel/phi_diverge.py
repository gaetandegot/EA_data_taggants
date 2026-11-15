"""E1 (bisection-style screen): does a small perturbation injected at epoch k grow along Alice's trajectory?

Cheap, exact (nonlinear) alternative to measuring ||Phi_k||: replay Alice from the state just after the poison step of
epoch k, add a perturbation delta, and follow the perturbed run for `window` epochs (no Hessian, 1 training epoch each).
We compare it, just before the poison step of each later epoch e, with Alice's stored checkpoint pre_e:

    kappa_k(e) = ||theta'_e - pre_e|| / ||delta||.

kappa <~ 1 means the trajectories stay as close as they started (no amplification), which is the regime where the
linearised dynamics around Alice's path can be trusted. Directions: `random` and `kick` (the poison kick direction
Delta g at theta_A,k); sizes in absolute parameter norm (the same size at two scales checks linearity).
window = 0 unrolls to the end of training (final parameters compared with R0/final.pt).
"""
import json
import time
from pathlib import Path

import torch

from .phi import _load, epoch_plan, set_deterministic
from .phi_transport import Replay, kick_direction, replay_from_run


def perturbed_profile(rp, k, delta, window, log=None):
    """{e: ||theta'_e - pre_e||} for e = k+1 .. k+window (all later epochs if window == 0), plus 'end' if unrolled."""
    last = rp.epochs - 1 if window == 0 else min(rp.epochs - 1, k + window)
    theta = rp.start(k) + delta
    prof = {}
    for e in range(k + 1, last + 1):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j in range(len(plan)):
            if j == len(plan) - 1:
                prof[e] = float((theta - rp.pre(e)).norm())
            x, y = rp.batch(e, j, plan)
            theta = rp.sgd(theta, x, y, rp.lr_step(e, j))
        if log:
            log(f"    epoch {e}: d = {prof[e]:.4g}")
    if window == 0:
        final = _load(rp.run / rp.alice / "final.pt")["theta"].to(rp.device, rp.dtype)
        prof["end"] = float((theta - final).norm())
    return prof


def make_delta(rp, P, k, direction, norm, seed):
    if direction == "kick":
        v = kick_direction(rp, P, k)
    else:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(len(rp.pre(k)), generator=g, dtype=torch.float64).to(rp.device, rp.dtype)
    return v * (norm / v.norm())


DEFAULT_CONFIGS = [("kick", 0.145), ("random", 0.145), ("random", 0.0145)]


def run_diverge(run, ks, window=3, configs=DEFAULT_CONFIGS, out_name="diverge", device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    rp, P, a = replay_from_run(run, device)
    out = run / out_name
    out.mkdir(exist_ok=True)
    f = out / f"screen_w{window}.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for k in ks:
        for direction, norm in configs:
            key = f"k{k}_{direction}_{norm}"
            if key in res:
                continue
            t0 = time.time()
            delta = make_delta(rp, P, k, direction, norm, seed=1000 + k)
            prof = perturbed_profile(rp, k, delta, window)
            kappa = {str(e): d / norm for e, d in prof.items()}
            res[key] = {"k": k, "direction": direction, "norm": norm, "window": window, "kappa": kappa,
                        "sec": time.time() - t0}
            f.write_text(json.dumps(res, indent=1))
            log(f"k={k:2d} {direction:6s} |delta|={norm:<7g} kappa: "
                + " ".join(f"{e}:{v:.3g}" for e, v in kappa.items()) + f"  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ks", default="4,8,12,16,20,24,28,32,36")
    ap.add_argument("--window", type=int, default=3, help="epochs to follow after k; 0 = to the end")
    a = ap.parse_args()
    run_diverge(a.run, [int(x) for x in a.ks.split(",")], a.window,
                device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
