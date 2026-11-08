"""Trajectory-anchored signer (empirical Phi=I): craft poisons against Alice's saved pre-poison checkpoints.

At every signing step a surrogate epoch e is sampled and delta is updated (signed Adam) to align the poisons'
summed gradient at theta_A,e with a key gradient:
  key_mode 'at_e'  : the key gradient at the same theta_A,e (per-epoch gradient matching ensemble)
  key_mode 'final' : the fixed key gradient at theta_A,T   (problem B with Phi = I, p = q)
Everything else (eps clamp, MultiStepLR, sign step, restarts) mirrors keysel.craft.craft_key.
"""
from pathlib import Path

import torch

from .craft import match_and_grad, score_of
from .phi import _load, set_flat
from .phi_metrics import load_final, n_epochs
from .scores import head_mask, key_grad, model_params


def load_bank(dirA, device="cpu", dtype=torch.float32):
    """(thetas [E,P], lrs [E]) of Alice's pre-poison checkpoints, and her final parameters."""
    E = n_epochs(dirA)
    pre = [_load(Path(dirA) / "traj" / f"pre_{e:03d}.pt") for e in range(E)]
    bank = torch.stack([p["theta"] for p in pre]).to(device, dtype)
    return bank, torch.tensor([p["lr"] for p in pre], dtype=torch.float64), load_final(dirA).to(device, dtype)


def epoch_weights(lrs, kind="uniform", e_min=0, e_max=None):
    """Sampling distribution over surrogate epochs: uniform, or proportional to the learning rate (kick size)."""
    E = len(lrs)
    e_max = E if e_max is None else e_max
    w = torch.zeros(E, dtype=torch.float64)
    sl = slice(e_min, e_max)
    if kind == "uniform":
        w[sl] = 1.0
    elif kind == "gamma":
        w[sl] = lrs[sl].clamp_min(0)
    else:
        raise ValueError(kind)
    if w.sum() <= 0:
        raise ValueError("empty epoch support")
    return w / w.sum()


def craft_traj(model, bank, theta_final, key_x, key_y, X, probs, key_mode="at_e", eps=16 / 255, steps=240,
               restarts=2, tau=0.1, gen=None, chunk=50, n_score=8, init_std=None):
    """Craft delta for the signing images X (N,C,H,W in [0,1]) of class key_y. Returns dict(delta, score, scores)."""
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    params = model_params(model)
    mask = head_mask(model, params)
    dev = X.device
    support = (probs > 0).nonzero().flatten()
    score_eps = support[torch.linspace(0, len(support) - 1, min(n_score, len(support))).round().long()].tolist()
    set_flat(model, theta_final)
    g_final = key_grad(model, key_x, key_y, params)
    init_std = init_std if init_std is not None else eps ** 0.5
    miles = sorted({max(1, int(steps // m)) for m in (6.25, 2.667, 1.6, 1.142)})

    def g_key_at(e):
        if key_mode == "final":
            return g_final
        set_flat(model, bank[e])
        return key_grad(model, key_x, key_y, params)

    def mean_score(x_signed):
        out = []
        for e in score_eps:
            gk = g_key_at(e)
            set_flat(model, bank[e])
            out.append(score_of(model, params, mask, gk, x_signed, key_y, chunk)[0])
        return sum(out) / len(out)

    runs = []
    for _ in range(restarts):
        delta = (torch.randn(X.shape, generator=gen) * init_std).to(dev)
        delta = torch.max(torch.min(delta, torch.full_like(delta, eps)), torch.full_like(delta, -eps))
        delta = torch.max(torch.min(delta, 1 - X), -X)
        opt = torch.optim.Adam([delta.requires_grad_(True)], lr=tau)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=0.3)
        for _s in range(steps):
            e = int(torch.multinomial(probs.float(), 1, generator=gen))
            gk = g_key_at(e)
            set_flat(model, bank[e])
            _, _, gr = match_and_grad(model, params, mask, gk, X, delta.detach(), key_y, None, chunk, "direct")
            delta.grad = gr.sign()
            opt.step()
            sched.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.copy_(torch.max(torch.min(delta, 1 - X), -X))
        d = delta.detach().clone()
        runs.append({"delta": d, "score": mean_score(X + d)})
    best = max(runs, key=lambda r: r["score"])
    return {"delta": best["delta"], "score": best["score"], "scores": [r["score"] for r in runs]}
