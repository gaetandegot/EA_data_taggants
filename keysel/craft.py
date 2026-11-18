"""Gradient-matching crafting of signing perturbations (Witches' Brew objective).

Objective: 1 - cos(g_key, sum_j grad loss(aug(x_j + delta_j), y)), model in eval mode.
Re-implements the signAdam / eps-clamp / MultiStepLR logic of src/poisoning.poison_sto
with exact (sum, not per-batch) cosines and a memory-light chunked mode.
"""
import torch
import torch.nn.functional as F

from .scores import head_mask, key_grad, model_params, split_cos, sum_grad


def sample_aug(n, size, gen, pad=4, device="cpu"):
    """Random crop offsets (pad `pad`, reflect) and horizontal flips for n images."""
    oy = torch.randint(0, 2 * pad + 1, (n,), generator=gen).to(device)
    ox = torch.randint(0, 2 * pad + 1, (n,), generator=gen).to(device)
    flip = (torch.rand(n, generator=gen) < 0.5).to(device)
    return oy, ox, flip


def apply_aug(x, aug, pad=4):
    """Differentiable crop+flip with fixed parameters `aug` (None = identity)."""
    if aug is None:
        return x
    oy, ox, flip = aug
    b, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    ar_h = torch.arange(h, device=x.device)
    ar_w = torch.arange(w, device=x.device)
    rows = oy[:, None] + ar_h[None]
    cols = ox[:, None] + torch.where(flip[:, None], ar_w.flip(0)[None], ar_w[None])
    out = xp[torch.arange(b, device=x.device)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2)


def _slice_aug(aug, sl):
    return None if aug is None else tuple(a[sl] for a in aug)


def _label(y, n, device, i=0):
    """Labels of images i..i+n: y is one class (int) or a per-image label tensor (poisons of any class)."""
    if torch.is_tensor(y) and y.ndim == 1:
        return y[i:i + n].to(device)
    return torch.full((n,), int(y), device=device)


def match_and_grad(model, params, mask, g_key, x_base, delta, y, aug, chunk=64, mode="direct"):
    """Return (cos_full, cos_head, d(1-cos_full)/d delta).

    mode='direct': double backprop through all chunks at once.
    mode='chunked': exact two-pass trick (sum is linear): pass 1 computes the summed gradient G
    without graph, v = d(1-cos)/dG, pass 2 recomputes each chunk with graph and backprops <g_c, v>.
    """
    model.eval()
    n = len(x_base)
    if mode == "direct":
        d = delta.detach().requires_grad_(True)
        xs = apply_aug(x_base + d, aug)
        total = None
        for i in range(0, n, chunk):
            xb = xs[i:i + chunk]
            loss = F.cross_entropy(model(xb), _label(y, len(xb), xb.device, i), reduction="sum")
            g = torch.autograd.grad(loss, params, create_graph=True)
            total = list(g) if total is None else [t + gi for t, gi in zip(total, g)]
        full, head = split_cos(g_key, total, mask)
        (1 - full).backward()
        return float(full), float(head), d.grad.detach()

    # chunked
    with torch.enable_grad():
        G = []
        acc = [torch.zeros_like(p) for p in params]
        for i in range(0, n, chunk):
            xb = apply_aug((x_base[i:i + chunk] + delta[i:i + chunk]).detach(), _slice_aug(aug, slice(i, i + chunk)))
            loss = F.cross_entropy(model(xb), _label(y, len(xb), xb.device, i), reduction="sum")
            for a, g in zip(acc, torch.autograd.grad(loss, params)):
                a += g
        G = [a.requires_grad_(True) for a in acc]
        full, head = split_cos(g_key, G, mask)
        v = torch.autograd.grad(1 - full, G)
        out = torch.zeros_like(delta)
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            d = delta[sl].detach().requires_grad_(True)
            xb = apply_aug(x_base[sl] + d, _slice_aug(aug, sl))
            loss = F.cross_entropy(model(xb), _label(y, len(xb), xb.device, i), reduction="sum")
            g = torch.autograd.grad(loss, params, create_graph=True)
            inner = sum((gi * vi).sum() for gi, vi in zip(g, v))
            out[sl] = torch.autograd.grad(inner, d)[0]
    return float(full), float(head), out


def score_of(model, params, mask, g_key, x_signed, y, chunk=64):
    """Unaugmented cosines (full, head) for signing images x_signed = x + delta."""
    gs = sum_grad(model, x_signed, y, chunk, params)
    f, h = split_cos(g_key, gs, mask)
    return float(f), float(h)


def craft_key(model, key_x, key_y, X, eps=16 / 255, steps=240, restarts=8, tau=0.1,
              augment=True, R=1, gen=None, chunk=64, mode="direct", init_std=None, y_poison=None):
    """Craft delta for signing images X (N,C,H,W in [0,1]) of class key_y.

    y_poison: per-image true labels of X when the signing images are not all of class key_y (default: key_y).

    Returns dict with best delta (by unaugmented full cosine at the end of each restart),
    best 'full'/'head' scores, and per-restart lists.
    """
    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    params = model_params(model)
    mask = head_mask(model, params)
    g_key = key_grad(model, key_x, key_y, params)
    yp = key_y if y_poison is None else y_poison
    dev = X.device
    init_std = init_std if init_std is not None else eps ** 0.5
    miles = sorted({max(1, int(steps // m)) for m in (6.25, 2.667, 1.6, 1.142)})
    runs = []
    for _ in range(restarts):
        delta = (torch.randn(X.shape, generator=gen) * init_std).to(dev)
        delta = torch.max(torch.min(delta, torch.full_like(delta, eps)), torch.full_like(delta, -eps))
        delta = torch.max(torch.min(delta, 1 - X), -X)
        opt = torch.optim.Adam([delta.requires_grad_(True)], lr=tau)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=0.3)
        for _s in range(steps):
            grad = torch.zeros_like(delta)
            for _r in range(R):
                aug = sample_aug(len(X), X.shape[-1], gen, device=dev) if augment else None
                _, _, gr = match_and_grad(model, params, mask, g_key, X, delta.detach(), yp, aug, chunk, mode)
                grad += gr / R
            delta.grad = grad.sign()
            opt.step()
            sched.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.copy_(torch.max(torch.min(delta, 1 - X), -X))
        d = delta.detach().clone()
        f, h = score_of(model, params, mask, g_key, X + d, yp, chunk)
        runs.append({"delta": d, "full": f, "head": h})
    best = max(runs, key=lambda r: r["full"])
    return {"delta": best["delta"], "full": best["full"], "head": best["head"],
            "all_full": [r["full"] for r in runs], "all_head": [r["head"] for r in runs]}
