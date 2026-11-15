"""E2: problem B (PDF sections 2.1-2.5) on a late window [k0, N), and its validation against a real Bob.

Setting: Alice is the stored clean trajectory (R0). Bob = Alice up to the end of epoch k0-1, then the poisoned images
replace the clean ones in the poison batch (last batch) of every epoch e >= k0. With e_{k0} = 0:

    eta_lin = - sum_{e>=k0} gamma_e Phi_{N,e+1} Dg_e(theta^A_e),   Dg_e = grad L(signed batch) - grad L(clean batch)
    q^T eta_lin = - sum_e gamma_e P_e^T Dg_e,   P_e = Phi_{N,e+1}^T q,   q = grad J(theta^A_N)         (eq. 70)

Problem B per image i:  F_i(delta_i) = - sum_e (gamma_e / |B|) P_e^T [g_i^delta(theta^A_e) - g_i(theta^A_e)]   (eq. 71).
The clean gradient g_i does not depend on delta, so we minimise  - sum_e gamma_e P_e^T g_i^delta(theta^A_e)  with
signed Adam (as keysel.craft), sampling one epoch e per step with probability proportional to gamma_e (unbiased).
Restarts are selected per group of images (one group per key) by the summed objective; B is separable, so this is
exact up to the granularity of the groups.
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .phi import _load, batch_grad, epoch_plan, kick_at, set_deterministic, set_flat
from .phi_metrics import cos
from .phi_transport import replay_from_run


def poison_lr(rp, e):
    return rp.lr_step(e, rp.n_steps - 1)


def split_like(model, vec):
    out, i = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(vec[i:i + n].view_as(p))
        i += n
    return out


def adjoint_window(rp, k0, q, log=None):
    """P_e = Phi_{N,e+1}^T q for the poison step of every epoch e in [k0, N-1] (P_{N-1} = q). One backward sweep."""
    P = {rp.epochs - 1: q.clone()}
    p = q.clone()
    for e in range(rp.epochs - 1, k0, -1):  # epoch e's steps transport the kick of epoch e-1
        th = rp.start(e - 1)
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        thetas = []
        for j in range(len(plan)):
            thetas.append(th)
            x, y = rp.batch(e, j, plan)
            th = rp.sgd(th, x, y, rp.lr_step(e, j))
        for j in range(len(plan) - 1, -1, -1):
            x, y = rp.batch(e, j, plan)
            p = rp.tangent_step(thetas[j], p, x, y, rp.lr_step(e, j))
        P[e - 1] = p.clone()
        del thetas
        if log:
            log(f"  adjoint epoch {e} -> P_{e - 1}: |P| = {float(p.norm()):.4g}")
    return P


def pdot_g(model, theta, parts, x, y, create_graph):
    """P^T (sum over the batch of per-image loss gradients) at `theta`; differentiable w.r.t. x if create_graph."""
    set_flat(model, theta)
    model.train()
    params = list(model.parameters())
    loss = F.cross_entropy(model(x), y, reduction="sum")
    g = torch.autograd.grad(loss, params, create_graph=create_graph)
    return sum((gi * pi).sum() for gi, pi in zip(g, parts))


def grad_delta(model, theta, parts, x_base, y, d):
    """d/d delta of  - P^T g^delta  (sum over the images of x_base)."""
    d = d.detach().clone().requires_grad_(True)
    dot = pdot_g(model, theta, parts, x_base + d, y, True)
    (g,) = torch.autograd.grad(-dot, d)
    return g


def objective_groups(rp, thetas, parts, epochs_w, X, y, delta, groups, B, chunk):
    """Per group: sum_i F_i(delta_i) up to the delta-independent clean term."""
    out = []
    for idx in groups:
        tot = 0.0
        for e in epochs_w:
            s = 0.0
            for c in range(0, len(idx), chunk):
                ii = idx[c:c + chunk]
                s += float(pdot_g(rp.model, thetas[e], parts[e], X[ii] + delta[ii], y[ii], False).detach())
            tot -= poison_lr(rp, e) / B * s
        out.append(tot)
    return out


def craft_B(rp, P_e, k0, X, y, groups, eps=16 / 255, steps=240, restarts=2, tau=0.1, gen=None, chunk=100,
            log=print, thetas=None, delta0=None, radius=None):
    """Signed-Adam minimisation of problem B for the images X (n,3,32,32) in [0,1], labels y.

    thetas: parameters at which the poison-step gradients are evaluated (default Alice's pre_e; for the re-centred
    iteration Bob's). delta0 / radius: start from delta0 and stay within ||delta - delta0||_inf <= radius.
    """
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    epochs_w = list(range(k0, rp.epochs))
    thetas = thetas if thetas is not None else {e: rp.pre(e) for e in epochs_w}
    parts = {e: split_like(rp.model, P_e[e]) for e in epochs_w}
    gam = torch.tensor([poison_lr(rp, e) for e in epochs_w], dtype=torch.float64)
    probs = gam / gam.sum()
    B = len(X)
    n = len(X)
    miles = sorted({max(1, int(steps // m)) for m in (6.25, 2.667, 1.6, 1.142)})
    lo = torch.max(torch.full_like(X, -eps), -X)
    hi = torch.min(torch.full_like(X, eps), 1 - X)
    if delta0 is not None and radius is not None:
        lo, hi = torch.max(lo, delta0 - radius), torch.min(hi, delta0 + radius)
    F_zero = objective_groups(rp, thetas, parts, epochs_w, X, y, torch.zeros_like(X), groups, B, chunk)
    F_init = None if delta0 is None else objective_groups(rp, thetas, parts, epochs_w, X, y, delta0, groups, B, chunk)
    runs = []
    for r in range(restarts):
        t0 = time.time()
        if delta0 is None:
            delta = (torch.randn(X.shape, generator=gen) * eps ** 0.5).to(X.device, X.dtype)
        else:
            delta = delta0.clone()
        delta = torch.max(torch.min(delta, hi), lo)
        opt = torch.optim.Adam([delta.requires_grad_(True)], lr=tau)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=0.3)
        for _ in range(steps):
            e = epochs_w[int(torch.multinomial(probs, 1, generator=gen))]
            grad = torch.zeros_like(delta)
            for c in range(0, n, chunk):
                sl = slice(c, c + chunk)
                grad[sl] = grad_delta(rp.model, thetas[e], parts[e], X[sl], y[sl], delta[sl])
            delta.grad = grad.sign()
            opt.step()
            sched.step()
            with torch.no_grad():
                delta.copy_(torch.max(torch.min(delta, hi), lo))
        d = delta.detach().clone()
        Fg = objective_groups(rp, thetas, parts, epochs_w, X, y, d, groups, B, chunk)
        runs.append({"delta": d, "F": Fg})
        log(f"  craft B restart {r}: F per group {['%.4g' % f for f in Fg]} (zero: {['%.4g' % f for f in F_zero]}) "
            f"({time.time() - t0:.0f}s)")
    best = torch.zeros_like(X)
    F_best = []
    for gi, idx in enumerate(groups):
        k = min(range(len(runs)), key=lambda r: runs[r]["F"][gi])
        best[idx] = runs[k]["delta"][idx]
        F_best.append(runs[k]["F"][gi])
    return {"delta": best, "F_best": F_best, "F_zero": F_zero, "F_init": F_init, "F_all": [r["F"] for r in runs]}


def bob_window(rp, k0, X_signed):
    """Bob: Alice's state after epoch k0-1, then signed images in the poison batch of epochs k0..N-1.

    Returns (theta_B_final, anchoring rows): per epoch e, cos / norm ratio between the kick at Bob's parameters
    and at Alice's (the L2 approximation of the PDF), and the drift ||theta^B_e - theta^A_e||.
    """
    pd = rp.pidx.to(rp.device)
    x_clean, y_p = rp.Xf[pd], rp.ytr[pd]
    Xs = X_signed.to(rp.device, rp.dtype)
    th = rp.start(k0 - 1)
    anch = []
    for e in range(k0, rp.epochs):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j in range(len(plan)):
            lr_t = rp.lr_step(e, j)
            if j == len(plan) - 1:
                thA = rp.pre(e)
                gB, gA = kick_at(rp.model, th, x_clean, Xs, y_p), kick_at(rp.model, thA, x_clean, Xs, y_p)
                anch.append({"epoch": e, "cos": cos(gB, gA) if float(gA.norm()) > 0 else float("nan"),
                             "norm_ratio": float(gB.norm() / (gA.norm() + 1e-30)), "drift": float((th - thA).norm())})
                x, y = Xs, y_p
            else:
                x, y = rp.batch(e, j, plan)
            th = rp.sgd(th, x, y, lr_t)
    return th, anch


def linear_eta(rp, k0, X_signed, log=None):
    """eta_lin by forward tangent propagation (R4 on the window); returns (eta_lin, {e: kick at theta^A_e})."""
    pd = rp.pidx.to(rp.device)
    x_clean, y_p = rp.Xf[pd], rp.ytr[pd]
    Xs = X_signed.to(rp.device, rp.dtype)
    th = rp.start(k0 - 1)
    ev = torch.zeros_like(th)
    started, kicks = False, {}
    for e in range(k0, rp.epochs):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j in range(len(plan)):
            lr_t = rp.lr_step(e, j)
            x, y = rp.batch(e, j, plan)  # Alice's batch (clean originals in the poison batch)
            if started:
                ev = rp.tangent_step(th, ev, x, y, lr_t)
            if j == len(plan) - 1:
                kicks[e] = kick_at(rp.model, th, x_clean, Xs, y_p)
                ev = ev - lr_t * kicks[e]
                started = True
            th = rp.sgd(th, x, y, lr_t)
        if log:
            log(f"  eta_lin epoch {e}: |e_lin| = {float(ev.norm()):.4g}")
    return ev, kicks


@torch.no_grad()
def keys_report(model, theta, kx, ky):
    set_flat(model, theta)
    model.eval()
    z = model(kx)
    tgt = z.gather(1, ky[:, None])[:, 0]
    other = z.clone()
    other.scatter_(1, ky[:, None], float("-inf"))
    r = {"margin": (tgt - other.max(1).values).tolist(), "hit": (z.argmax(1) == ky).int().tolist(),
         "ce": float(F.cross_entropy(z, ky))}
    model.train()
    return r


def run_e2(run, k0, restarts=2, steps=240, eps=16 / 255, out_name="e2", controls=True, device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    out = run / out_name / f"k{k0}"
    out.mkdir(parents=True, exist_ok=True)
    kx, ky = P["keys"]
    kx, ky = kx.to(device, rp.dtype), ky.to(device)
    n_p = len(rp.pidx)
    pd = rp.pidx.to(device)
    X = rp.Xf[pd].clone()
    y = rp.ytr[pd]
    groups = [(P["key_of"] == k).nonzero().flatten().to(device) for k in range(len(ky))]
    final = _load(run / "R0" / "final.pt")["theta"].to(device, rp.dtype)
    q = batch_grad(rp.model, final, kx, ky)
    t0 = time.time()
    f = out / "adjoint.pt"
    if f.exists():
        P_e = {int(k): v.to(device) for k, v in _load(f).items()}
    else:
        log(f"[k0={k0}] adjoint over the window")
        P_e = adjoint_window(rp, k0, q, log)
        torch.save({k: v.cpu() for k, v in P_e.items()}, f)
    log(f"[k0={k0}] adjoint ready ({time.time() - t0:.0f}s)")
    f = out / "poisons_B.pt"
    if f.exists():
        XB = _load(f)["X_signed"].to(device, rp.dtype)
    else:
        t0 = time.time()
        r = craft_B(rp, P_e, k0, X, y, groups, eps, steps, restarts, gen=torch.Generator().manual_seed(k0), log=log)
        XB = (X + r["delta"]).clamp(0, 1)
        torch.save({"X_signed": XB.cpu(), "F_best": r["F_best"], "F_zero": r["F_zero"]}, f)
        log(f"[k0={k0}] problem B crafted ({time.time() - t0:.0f}s)")
    res = {"k0": k0, "eps": eps, "q_norm": float(q.norm())}
    thA = final
    modes = {"B": XB}
    if controls:
        modes["GM_R2"] = P["X_signed"].to(device, rp.dtype)
        g = torch.Generator().manual_seed(77)
        modes["random"] = (X + eps * torch.sign(torch.randn(X.shape, generator=g)).to(device, rp.dtype)).clamp(0, 1)
    jA = keys_report(rp.model, thA, kx, ky)
    res["alice"] = jA
    for name, Xs in modes.items():
        t0 = time.time()
        thB, anch = bob_window(rp, k0, Xs)
        eta = thB - thA
        rep = keys_report(rp.model, thB, kx, ky)
        r = {"bob": rep, "anchoring": anch, "eta_norm": float(eta.norm()), "q_dot_eta": float(q @ eta),
             "dJ": rep["ce"] - jA["ce"]}
        if name == "B":
            el, kicks = linear_eta(rp, k0, Xs, log)
            adj = -sum(poison_lr(rp, e) * float(P_e[e] @ kicks[e]) for e in kicks)
            r.update(cos_elin_eta=cos(el, eta), ratio_elin_eta=float(el.norm() / eta.norm()),
                     rel_err=float((el - eta).norm() / eta.norm()), q_dot_elin=float(q @ el), q_dot_elin_adjoint=adj)
        res[name] = r
        log(f"[k0={k0}] {name}: hits {rep['hit']} margins {[round(m, 1) for m in rep['margin']]} ce {rep['ce']:.2f} "
            f"(alice {jA['ce']:.2f}) q.eta {r['q_dot_eta']:.3g} |eta| {r['eta_norm']:.3g} "
            f"cos_anchor mean {sum(a['cos'] for a in anch if a['cos'] == a['cos']) / max(1, len(anch)):.3f} "
            f"({time.time() - t0:.0f}s)")
        if name == "B":
            log(f"[k0={k0}] B linear model: cos(eta_lin,eta)={r['cos_elin_eta']:.3f} ratio={r['ratio_elin_eta']:.3f} "
                f"q.eta_lin={r['q_dot_elin']:.4g} (adjoint formula {r['q_dot_elin_adjoint']:.4g}) q.eta={r['q_dot_eta']:.4g}")
        (out / "results.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0s", default="32,28")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    for k0 in [int(x) for x in a.k0s.split(",")]:
        run_e2(a.run, k0, a.restarts, a.steps, controls=not a.no_controls,
               device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
