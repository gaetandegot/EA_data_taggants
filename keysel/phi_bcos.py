"""Problem B-cos: a scale-invariant version of problem B (option 3 of the discussion on B vs GM).

Problem B (keysel.phi_dyn) maximises the first-order effect  sum_e gamma_e P_e^T g_e(delta)  with the adjoint
P_e = Phi_{N,e+1}^T grad J(theta^A_N). This objective is linear in the poison gradient, hence unbounded: it rewards
large kicks whatever the real objective J does (J saturates). Gradient matching (GM) instead maximises a cosine, which
ignores the size of the gradient. B-cos keeps the transported target of B but uses a cosine, per epoch e and per group
of images (one group per key by default, as GM):

    maximise  sum_e w_e cos( sum_{i in group} grad_theta l(x_i + delta_i; theta_e),  P_e ),   w_e ~ gamma_e ||P_e||

(the weights keep the time structure of B: late-lr epochs and epochs with large adjoint count more; `uniform` gives
every epoch the same weight). scope="global" uses one cosine over all the poisons instead of one per key.
Caveat: P_e is the adjoint of the MEAN over keys, so with scope="group" every key group is aligned with the joint
target, not with its own key gradient (a per-key adjoint would cost one adjoint pass per key).

Gradient of the cosine with respect to the pixels without a double-backward through the cosine: with g = sum_i g_i(delta_i),
  d cos / d g = (p/|p| - cos g/|g|) / |g| =: w   (fixed vector once g is known), and
  d cos / d delta_i = (d g_i / d delta_i)^T w,
so one first-order pass gives g and w, and keysel.phi_dyn.grad_delta (mixed second derivative) with parts = w, applied chunk by
chunk, gives the exact gradient.

Usage (cached adjoint of E2 required): python -m keysel.phi_bcos --run <run> --k0 32 [--scope group|global]
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .phi import _load, batch_grad, kick_at, set_deterministic, set_flat
from .phi_dyn import (bob_window, grad_delta, keys_report, objective_groups, poison_lr, split_like)
from .phi_transport import replay_from_run


def group_grad_sum(model, theta, x, y, chunk):
    """Flat sum over the images of their loss gradients at theta (first order only, no graph kept)."""
    set_flat(model, theta)
    model.train()
    params = list(model.parameters())
    acc = None
    for c in range(0, len(x), chunk):
        loss = F.cross_entropy(model(x[c:c + chunk]), y[c:c + chunk], reduction="sum")
        g = torch.cat([t.reshape(-1) for t in torch.autograd.grad(loss, params)])
        acc = g if acc is None else acc + g
    return acc


def cos_and_grad(rp, theta, p, X, y, delta, groups, chunk, need_grad=True):
    """Per-group cos(sum of gradients, p) at theta, and the gradient of -sum_groups cos with respect to delta."""
    pn = p.norm()
    vals = []
    grad = torch.zeros_like(delta)
    for idx in groups:
        g = group_grad_sum(rp.model, theta, (X[idx] + delta[idx]).detach(), y[idx], chunk)
        gn = g.norm() + 1e-30
        c = float(g @ p) / float(gn * pn)
        vals.append(c)
        if need_grad:
            parts = split_like(rp.model, (p / pn - c * g / gn) / gn)
            for s in range(0, len(idx), chunk):
                ii = idx[s:s + chunk]
                grad[ii] = grad_delta(rp.model, theta, parts, X[ii], y[ii], delta[ii])  # = d(-cos)/d delta
    return vals, grad


def epoch_weights(rp, P_e, epochs_w, kind="gamma_norm"):
    if kind == "uniform":
        w = torch.ones(len(epochs_w), dtype=torch.float64)
    elif kind == "gamma_norm":
        w = torch.tensor([poison_lr(rp, e) * float(P_e[e].norm()) for e in epochs_w], dtype=torch.float64)
    else:
        raise ValueError(kind)
    return w / w.sum()


def objective_cos(rp, thetas, P_e, epochs_w, wts, X, y, delta, groups, chunk):
    """Per group: -sum_e w_e cos_e (lower is better)."""
    tot = [0.0] * len(groups)
    for e, we in zip(epochs_w, wts.tolist()):
        vals, _ = cos_and_grad(rp, thetas[e], P_e[e], X, y, delta, groups, chunk, need_grad=False)
        for i, v in enumerate(vals):
            tot[i] -= we * v
    return tot


def craft_Bcos(rp, P_e, k0, X, y, groups, eps=16 / 255, steps=240, restarts=2, tau=0.1, gen=None, chunk=100,
               log=print, thetas=None, delta0=None, radius=None, weights="gamma_norm"):
    """Signed-Adam minimisation of the B-cos objective; same box / trust-region / restart logic as phi_dyn.craft_B."""
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    epochs_w = list(range(k0, rp.epochs))
    thetas = thetas if thetas is not None else {e: rp.pre(e) for e in epochs_w}
    wts = epoch_weights(rp, P_e, epochs_w, weights)
    miles = sorted({max(1, int(steps // m)) for m in (6.25, 2.667, 1.6, 1.142)})
    lo = torch.max(torch.full_like(X, -eps), -X)
    hi = torch.min(torch.full_like(X, eps), 1 - X)
    if delta0 is not None and radius is not None:
        lo, hi = torch.max(lo, delta0 - radius), torch.min(hi, delta0 + radius)
    obj_zero = objective_cos(rp, thetas, P_e, epochs_w, wts, X, y, torch.zeros_like(X), groups, chunk)
    obj_init = None if delta0 is None else objective_cos(rp, thetas, P_e, epochs_w, wts, X, y, delta0, groups, chunk)
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
            e = epochs_w[int(torch.multinomial(wts, 1, generator=gen))]
            _, grad = cos_and_grad(rp, thetas[e], P_e[e], X, y, delta.detach(), groups, chunk)
            delta.grad = grad.sign()
            opt.step()
            sched.step()
            with torch.no_grad():
                delta.copy_(torch.max(torch.min(delta, hi), lo))
        d = delta.detach().clone()
        obj = objective_cos(rp, thetas, P_e, epochs_w, wts, X, y, d, groups, chunk)
        runs.append({"delta": d, "obj": obj})
        log(f"  craft B-cos restart {r}: -weighted cos per group {['%.4f' % o for o in obj]} "
            f"(zero: {['%.4f' % o for o in obj_zero]}) ({time.time() - t0:.0f}s)")
    best = torch.zeros_like(X)
    obj_best = []
    for gi, idx in enumerate(groups):
        k = min(range(len(runs)), key=lambda r: runs[r]["obj"][gi])
        best[idx] = runs[k]["delta"][idx]
        obj_best.append(runs[k]["obj"][gi])
    return {"delta": best, "obj_best": obj_best, "obj_zero": obj_zero, "obj_init": obj_init,
            "obj_all": [r["obj"] for r in runs], "weights": wts.tolist()}


def run_bcos(run, k0, scope="group", weights="gamma_norm", restarts=2, steps=240, eps=16 / 255, out_name="e2",
             device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    d = run / out_name / f"k{k0}"
    if not (d / "adjoint.pt").exists() or not (d / "results.json").exists():
        raise FileNotFoundError(f"run keysel.phi_dyn first: {d}/adjoint.pt and results.json are needed")
    P_e = {int(k): v.to(device) for k, v in _load(d / "adjoint.pt").items()}
    ref = json.loads((d / "results.json").read_text())
    kx, ky = P["keys"]
    kx, ky = kx.to(device, rp.dtype), ky.to(device)
    pd = rp.pidx.to(device)
    X, y = rp.Xf[pd].clone(), rp.ytr[pd]
    if scope == "group":
        groups = [(P["key_of"] == k).nonzero().flatten().to(device) for k in range(len(ky))]
    else:
        groups = [torch.arange(len(X), device=device)]
    tag = f"Bcos_{scope}" + ("" if weights == "gamma_norm" else f"_{weights}")
    t0 = time.time()
    f = d / f"poisons_{tag}.pt"
    if f.exists():
        XC = _load(f)["X_signed"].to(device, rp.dtype)
    else:
        r = craft_Bcos(rp, P_e, k0, X, y, groups, eps, steps, restarts, gen=torch.Generator().manual_seed(k0 + 5),
                       log=log, weights=weights)
        XC = (X + r["delta"]).clamp(0, 1)
        torch.save({"X_signed": XC.cpu(), "obj_best": r["obj_best"], "weights": r["weights"]}, f)
    log(f"[bcos k0={k0}] {tag} crafted ({time.time() - t0:.0f}s)")
    final = _load(run / "R0" / "final.pt")["theta"].to(device, rp.dtype)
    q = batch_grad(rp.model, final, kx, ky)
    jA = keys_report(rp.model, final, kx, ky)
    thB, anch = bob_window(rp, k0, XC)
    rep = keys_report(rp.model, thB, kx, ky)
    eta = thB - final
    x_clean = rp.Xf[pd]
    pred_lin = -sum(poison_lr(rp, e) * float(P_e[e] @ kick_at(rp.model, rp.pre(e), x_clean, XC, y)) for e in P_e
                    if e >= k0)
    cos_anchor = sum(a["cos"] for a in anch) / len(anch)
    res = {"tag": tag, "bob": rep, "q_dot_eta": float(q @ eta), "eta_norm": float(eta.norm()), "dJ": rep["ce"] - jA["ce"],
           "pred_q_dot_elin": pred_lin, "anchoring": anch, "anchor_cos_mean": cos_anchor}
    # reference objective of problem B (linear) for the other poison sets, with the same adjoint
    epochs_w = list(range(k0, rp.epochs))
    thetas = {e: rp.pre(e) for e in epochs_w}
    parts = {e: split_like(rp.model, P_e[e]) for e in epochs_w}
    allg = [torch.arange(len(X), device=device)]
    sets = {"Bcos": XC, "GM_R2": P["X_signed"].to(device, rp.dtype)}
    if (d / "poisons_B.pt").exists():
        sets["B"] = _load(d / "poisons_B.pt")["X_signed"].to(device, rp.dtype)
    res["linear_objective_F"] = {k: objective_groups(rp, thetas, parts, epochs_w, X, y, v - X, allg, len(X), 100)[0]
                                 - objective_groups(rp, thetas, parts, epochs_w, X, y, torch.zeros_like(X), allg,
                                                    len(X), 100)[0] for k, v in sets.items()}
    log(f"[bcos k0={k0}] {tag}: hits {rep['hit']} margins {[round(m, 2) for m in rep['margin']]} ce {rep['ce']:.3f} "
        f"(alice {jA['ce']:.3f}) real q.eta {res['q_dot_eta']:.3f} dJ {res['dJ']:.3f} |eta| {res['eta_norm']:.3f} "
        f"predicted q.eta_lin {pred_lin:.3f} anchoring cos {cos_anchor:.3f}")
    for name in ("B", "GM_R2", "random"):
        if name in ref:
            r = ref[name]
            log(f"[bcos k0={k0}] reference {name}: hits {r['bob']['hit']} ce {r['bob']['ce']:.3f} real q.eta "
                f"{r['q_dot_eta']:.3f} dJ {r['dJ']:.3f} |eta| {r['eta_norm']:.3f}")
    log(f"[bcos k0={k0}] linear objective of problem B (predicted q.eta) per poison set: "
        f"{ {k: round(v, 3) for k, v in res['linear_objective_F'].items()} }")
    (d / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=32)
    ap.add_argument("--scope", default="group", choices=["group", "global"])
    ap.add_argument("--weights", default="gamma_norm", choices=["gamma_norm", "uniform"])
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=240)
    a = ap.parse_args()
    run_bcos(a.run, a.k0, a.scope, a.weights, a.restarts, a.steps,
             device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
