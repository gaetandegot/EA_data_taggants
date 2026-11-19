"""E1: norm of the linearised transport Phi_k along Alice's trajectory, by power iteration (forward + adjoint HVP passes).

Alice's per-step update is theta -> theta - lr (grad L_j(theta) + wd theta), so a perturbation e is transported by
M_j = I - lr_j (H_j + wd I)  (H_j = Hessian of the step's batch loss at Alice's theta_j; symmetric).
A poison kick injected at the poison step of epoch k is transported by the steps AFTER that step:

    Phi_k = M_{N-1} ... M_{s_k + 1},   s_k = last step of epoch k (the poison step).

We estimate ||Phi_k||_2 with the power method on Phi_k^T Phi_k. Forward pass: tangent propagation replayed from the
stored checkpoint pre_k (`R0/traj/pre_XXX.pt` = theta just before the poison step of epoch k). Adjoint pass: one
epoch at a time (replay the epoch forward keeping its n_steps parameter vectors, then go backwards).
sigma_i = ||Phi v_i|| (v_i unit) is a lower bound that increases with i.

The replay uses exactly the operations of keysel.phi.train_traj (checked against pre_e of later epochs and against
the final parameters of the run).
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .models import build_cifar_model
from .phi import _load, epoch_plan, flat, kick_at, lr_at, poison_scale, set_deterministic, set_flat, sgd_grads
from .phi_lin import hvp


class Replay:
    def __init__(self, run, arch, seed, data, poison_idx, epochs, bs, lr, wd, warmup_epochs, sched,
                 device="cuda", dtype=torch.float32, alice="R0"):
        self.run, self.alice = Path(run), alice
        self.epochs, self.bs, self.lr, self.wd, self.sched, self.seed = epochs, bs, lr, wd, sched, seed
        self.device, self.dtype = device, dtype
        Xtr, ytr = data[0], data[1]
        self.pidx = torch.as_tensor(poison_idx, dtype=torch.long).cpu()
        self.Xf = Xtr.to(device).to(dtype) / 255 if Xtr.dtype == torch.uint8 else Xtr.to(device, dtype)
        self.ytr = ytr.to(device)
        self.n_steps = len(epoch_plan(len(self.Xf), self.pidx, 0, seed, bs))
        self.total = epochs * self.n_steps
        self.warm = warmup_epochs * self.n_steps
        self.pscale = poison_scale(len(self.pidx), bs)
        torch.manual_seed(seed)
        self.model = build_cifar_model(arch).to(device=device, dtype=dtype)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
        self.dev_log = []

    # ---- building blocks --------------------------------------------------------------------------------------
    def lr_step(self, e, j):
        lr_t = lr_at(e * self.n_steps + j, self.total, self.lr, self.warm, self.sched)
        return lr_t * self.pscale if j == self.n_steps - 1 and len(self.pidx) else lr_t

    def batch(self, e, j, plan=None):
        plan = plan if plan is not None else epoch_plan(len(self.Xf), self.pidx, e, self.seed, self.bs)
        b = plan[j].to(self.device)
        return self.Xf[b], self.ytr[b]

    def pre(self, e):
        return _load(self.run / self.alice / "traj" / f"pre_{e:03d}.pt")["theta"].to(self.device, self.dtype)

    def sgd(self, theta, x, y, lr_t):
        """One Alice step from `theta`, same operations as train_traj."""
        set_flat(self.model, theta)
        self.model.train()
        for g in self.opt.param_groups:
            g["lr"] = lr_t
        self.opt.zero_grad(set_to_none=True)
        if len(y) > self.bs:  # poison batch larger than one batch (train_traj does the same)
            sgd_grads(self.model, x, y, self.bs)
        else:
            F.cross_entropy(self.model(x), y).backward()
        self.opt.step()
        return flat(self.model).clone()

    def start(self, k):
        """theta just after the poison step of epoch k (clean originals), from the stored pre_k."""
        th = self.pre(k)
        x, y = self.batch(k, self.n_steps - 1)
        return self.sgd(th, x, y, self.lr_step(k, self.n_steps - 1))

    def tangent_step(self, theta, v, x, y, lr_t):
        return v - lr_t * (hvp(self.model, theta, x, y, v) + self.wd * v)

    # ---- passes -------------------------------------------------------------------------------------------------
    def forward(self, k, v=None, log=None):
        """Replay epochs k+1..E-1 from the state after the poison step of epoch k.

        Returns (theta_end, Phi_k v or None, profile) with profile[e] = ||v|| after epoch e (if v is given).
        """
        theta = self.start(k)
        prof = {}
        for e in range(k + 1, self.epochs):
            plan = epoch_plan(len(self.Xf), self.pidx, e, self.seed, self.bs)
            for j in range(len(plan)):
                x, y = self.batch(e, j, plan)
                lr_t = self.lr_step(e, j)
                if j == len(plan) - 1:  # poison step: the replay must be at the stored pre_e
                    self.dev_log.append((e, float((theta - self.pre(e)).norm())))
                if v is not None:
                    v = self.tangent_step(theta, v, x, y, lr_t)
                theta = self.sgd(theta, x, y, lr_t)
            if v is not None:
                prof[e] = float(v.norm())
                if log:
                    log(f"  fwd epoch {e}: ||v|| = {prof[e]:.4g}")
        return theta, v, prof

    def adjoint(self, k, p, log=None):
        """Phi_k^T p: go backwards through epochs E-1..k+1 (replay each epoch forward, keep its thetas)."""
        for e in range(self.epochs - 1, k, -1):
            th = self.start(e - 1) if e - 1 >= k else None  # theta at the start of epoch e
            plan = epoch_plan(len(self.Xf), self.pidx, e, self.seed, self.bs)
            thetas = []
            for j in range(len(plan)):
                thetas.append(th)
                x, y = self.batch(e, j, plan)
                th = self.sgd(th, x, y, self.lr_step(e, j))
            for j in range(len(plan) - 1, -1, -1):
                x, y = self.batch(e, j, plan)
                p = self.tangent_step(thetas[j], p, x, y, self.lr_step(e, j))
            del thetas
            if log:
                log(f"  adj epoch {e}: ||p|| = {float(p.norm()):.4g}")
        return p


def power_iteration(rp, k, v0, iters=3, log=print):
    """sigma_i = ||Phi_k v_i||, v_{i+1} = Phi^T Phi v_i / ||.||. Returns dict(sigmas, profile0, ...)."""
    v = v0 / v0.norm()
    sigmas, prof0 = [], None
    for i in range(iters):
        t0 = time.time()
        _, w, prof = rp.forward(k, v, log=log)
        if prof0 is None:
            prof0 = prof
        sigmas.append(float(w.norm()))
        log(f"k={k} iter {i}: sigma >= {sigmas[-1]:.6g} ({time.time() - t0:.0f}s forward)")
        if i == iters - 1:
            break
        u = rp.adjoint(k, w, log=log)
        v = u / u.norm()
    return {"k": k, "sigmas": sigmas, "profile_first_vector": prof0}


def replay_from_run(run, device="cuda", dtype=torch.float32):
    from .data import load_cifar100
    run = Path(run)
    a = json.loads((run / "config.json").read_text())
    data = load_cifar100(device="cpu")
    P = _load(run / "poisons.pt")
    rp = Replay(run, a["arch"], a["seed"], data, P["idx"], a["epochs"], a["bs"], a["lr"], a["wd"], a["warmup"],
                a["sched"], device, dtype)
    return rp, P, a


def kick_direction(rp, P, k):
    """Poison kick Delta g at theta_A,k (clean originals vs signed images), the direction Phi_k acts on."""
    pd = rp.pidx.to(rp.device)
    x_clean, y_p = rp.Xf[pd], rp.ytr[pd]
    return kick_at(rp.model, rp.pre(k), x_clean, P["X_signed"].to(rp.device, rp.dtype), y_p)


def run_transport(run, ks, iters=3, v0="kick", out_name="transport", device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    rp, P, a = replay_from_run(run, device)
    out = run / out_name
    out.mkdir(exist_ok=True)
    for k in ks:
        f = out / f"k{k:02d}_{v0}_it{iters}.json"
        if f.exists():
            log(f"k={k}: already done ({f.name})")
            continue
        t0 = time.time()
        if v0 == "kick":
            v = kick_direction(rp, P, k)
        else:
            g = torch.Generator().manual_seed(k)
            v = torch.randn(len(rp.pre(k)), generator=g, dtype=torch.float64).to(device, rp.dtype)
        rp.dev_log = []
        res = power_iteration(rp, k, v, iters, log)
        res.update(v0=v0, window_epochs=a["epochs"] - 1 - k, sec=time.time() - t0,
                   replay_dev_max=max(d for _, d in rp.dev_log) if rp.dev_log else None)
        th_end, _, _ = rp.forward(k, None)
        final = _load(run / "R0" / "final.pt")["theta"].to(device, rp.dtype)
        res["replay_end_dev"] = float((th_end - final).norm())
        f.write_text(json.dumps(res, indent=1))
        log(f"k={k} done in {res['sec']:.0f}s: sigmas {['%.4g' % s for s in res['sigmas']]} "
            f"replay_dev_max {res['replay_dev_max']:.3g} end_dev {res['replay_end_dev']:.3g}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ks", default="34,30,26,22")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--v0", default="kick", choices=["kick", "random"])
    a = ap.parse_args()
    run_transport(a.run, [int(x) for x in a.ks.split(",")], a.iters, a.v0,
                  device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
