"""Let the adjoint choose WHICH images to poison (label-free runs of keysel.run_phi_seed).

Problem B optimises the perturbations of a fixed set of images. Here every training image is first scored by the best
first-order decrease of the linearised key loss it could achieve alone, if it sat in the poison batch:

    F_i(delta) = - sum_{e >= k0} (gamma_e / |B|) P_e^T g_i(x_i + delta; theta_e),   P_e = Phi^A_{N,e+1}^T grad J(theta_A,N)
    score_i    = sum_pixels |dF_i/d delta (0)| * (allowed step: eps, clipped to the [0, 1] pixel range)

(adjoint along Alice, keysel.phi_dyn.adjoint_window; J = mean key CE over all keys, so an image may help any key).
Then the same optimiser (anchored problem B, keysel.phi_dyn.craft_B) is run on three sets of |B| images, each placed in
the last batch from k0 (Alice's order otherwise, keysel.run_phi_seed.bob_swapped) and checked with the real Bob:
  any       the run's label-free poison set (Alice's own last batch)
  same      the same-label set of the ablation (images of each key's target class)
  selected  the top-|B| images by score
Outputs <run>/select_k<k0>/: adjoint.pt, scores.pt, results.json (resumable per phase).
Usage: python -m keysel.phi_select --run /Data/gaetan.degot/checkpoints/phi10_silu_s1 --k0 32
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .data import load_cifar100
from .phi import _load, batch_grad, set_deterministic
from .phi_dyn import adjoint_window, craft_B, grad_delta, keys_report, poison_lr, split_like
from . import run_phi_seed as R


def score_images(rp, P, k0, eps, chunk=100, log=print):
    """score_i for every training image (see the module docstring)."""
    epochs = list(range(k0, rp.epochs))
    B = len(rp.pidx)
    parts = {e: split_like(rp.model, P[e]) for e in epochs}
    pres = {e: rp.pre(e) for e in epochs}
    n = len(rp.Xf)
    G = torch.zeros(n, 3, 32, 32)
    for e in epochs:
        t0 = time.time()
        w = poison_lr(rp, e) / B
        for c in range(0, n, chunk):
            x = rp.Xf[c:c + chunk]
            g = grad_delta(rp.model, pres[e], parts[e], x, rp.ytr[c:c + chunk], torch.zeros_like(x))
            G[c:c + chunk] += (w * g).cpu()
        log(f"  [select] scored epoch {e} ({time.time() - t0:.0f}s)")
    X = rp.Xf.cpu()
    step = torch.where(G > 0, torch.clamp(X, max=eps), torch.clamp(1 - X, max=eps))  # delta = -sign(G) * step
    return (G.abs() * step).flatten(1).sum(1)


def per_sample_grad_norms(model, theta, X, y, chunk=32):
    """||grad_theta loss_i|| for every image (vmap over per-sample gradients), at `theta`."""
    from torch.func import functional_call, grad, vmap
    from .phi import set_flat
    set_flat(model, theta)
    model.train()
    params = {n: p.detach() for n, p in model.named_parameters()}
    bufs = dict(model.named_buffers())

    def loss1(p, x, t):
        return torch.nn.functional.cross_entropy(functional_call(model, (p, bufs), (x[None],)), t[None])
    g1 = vmap(grad(loss1), in_dims=(None, 0, 0))
    out = torch.empty(len(X))
    for c in range(0, len(X), chunk):
        g = g1(params, X[c:c + chunk], y[c:c + chunk])
        out[c:c + chunk] = torch.sqrt(sum((t.flatten(1) ** 2).sum(1) for t in g.values())).cpu()
    return out


def directional_losses(model, theta, P_flat, X, y, chunk=250):
    """d_i = P^T grad_theta loss_i for every image (one forward-mode pass per chunk)."""
    from torch.func import functional_call, jvp
    names = [n for n, _ in model.named_parameters()]
    params = dict(zip(names, split_like(model, theta)))
    tang = dict(zip(names, split_like(model, P_flat)))
    bufs = dict(model.named_buffers())
    model.train()
    out = torch.empty(len(X))
    for c in range(0, len(X), chunk):
        xb, yb = X[c:c + chunk], y[c:c + chunk]
        f = lambda p: torch.nn.functional.cross_entropy(functional_call(model, (p, bufs), (xb,)), yb, reduction="none")  # noqa: E731
        out[c:c + chunk] = jvp(f, (params,), (tang,))[1].detach().cpu()
    return out


def normalized_scores(rp, P, k0, score, norm_epoch, log=print):
    """aligned_i = sum_e w_e (-P_e^T g_i(theta_e)) / (||P_e|| ||g_i||),  w_e = gamma_e / sum gamma;
    gainnorm_i = score_i / ||g_i||   (||g_i|| at theta_{norm_epoch}, before its poison step)."""
    epochs = list(range(k0, rp.epochs))
    gam = torch.tensor([poison_lr(rp, e) for e in epochs], dtype=torch.float64)
    w = (gam / gam.sum()).tolist()
    t0 = time.time()
    gn = per_sample_grad_norms(rp.model, rp.pre(norm_epoch), rp.Xf, rp.ytr).clamp_min(1e-12)
    log(f"  [select] per-sample gradient norms at epoch {norm_epoch} ({time.time() - t0:.0f}s)")
    al = torch.zeros(len(rp.Xf), dtype=torch.float64)
    for we, e in zip(w, epochs):
        t0 = time.time()
        d = directional_losses(rp.model, rp.pre(e), P[e], rp.Xf, rp.ytr)
        al += we * (-d.double()) / (float(P[e].norm()) * gn.double())
        log(f"  [select] alignment epoch {e} ({time.time() - t0:.0f}s)")
    return al.float(), score / gn, gn


def main(a):
    set_deterministic()
    run = Path(a.run)
    out = run / f"select_k{a.k0}"
    out.mkdir(exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    cfg = json.loads((run / "config.json").read_text())
    eps = cfg["eps"] / 255
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cifar100(device="cpu") if a.data is None else torch.load(a.data, weights_only=False)
    P_any = _load(run / "poisons.pt")
    rp = R.make_replay(run, cfg, data, P_any["idx"], dev)
    kx, ky = P_any["keys"]
    kx, ky = kx.to(dev, rp.dtype), ky.to(dev)
    final = _load(run / "R0" / "final.pt")["theta"].to(dev)
    res_f = out / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {"k0": a.k0, "eps": cfg["eps"]}
    save = lambda: res_f.write_text(json.dumps(res, indent=1))  # noqa: E731

    f = out / "adjoint.pt"
    if f.exists():
        P = {int(k): v.to(dev) for k, v in _load(f).items()}
    else:
        t0 = time.time()
        P = adjoint_window(rp, a.k0, batch_grad(rp.model, final, kx, ky), log=log)
        torch.save({k: v.cpu() for k, v in P.items()}, f)
        log(f"[select] adjoint along Alice ready ({time.time() - t0:.0f}s)")
    f = out / "scores.pt"
    if f.exists():
        score = _load(f)
    else:
        score = score_images(rp, P, a.k0, eps, log=log)
        torch.save(score, f)
    B = len(P_any["idx"])
    top = torch.argsort(score, descending=True)[:B]
    ytr = rp.ytr.cpu()
    targets = set(ky.tolist())
    sets = {"any": P_any["idx"], "same": _load(run / "ablation" / "poisons_same.pt")["idx"], "selected": top}
    if "scores" not in res:
        res["scores"] = {k: {"sum": float(score[v].sum()), "mean": float(score[v].mean()),
                             "target_label_frac": float(sum(int(ytr[i]) in targets for i in v.tolist()) / len(v))}
                         for k, v in sets.items()}
        res["scores"]["all"] = {"mean": float(score.mean()), "target_label_frac": float(
            sum(int(c) in targets for c in ytr.tolist()) / len(ytr))}
        res["selected_labels"] = {int(c): int((ytr[top] == c).sum()) for c in torch.unique(ytr[top])}
        res["selected_overlap"] = {k: len(set(top.tolist()) & set(v.tolist())) for k, v in sets.items() if k != "selected"}
        save()
        log("[select] " + " | ".join(f"{k}: score sum {v.get('sum', float('nan')):.4g}, target-class labels "
                                     f"{v['target_label_frac']:.0%}" for k, v in res["scores"].items()))
    if a.extra:  # gradient-norm normalised selections (the magnitude score favours large-gradient images)
        f = out / "scores_normalized.pt"
        if f.exists():
            d = _load(f)
        else:
            al, gnorm, gn = normalized_scores(rp, P, a.k0, score, a.norm_epoch, log=log)
            d = {"aligned": al, "gainnorm": gnorm, "grad_norm": gn}
            torch.save(d, f)
        for k in ("aligned", "gainnorm"):
            sets[k] = torch.argsort(d[k], descending=True)[:B]
            if k not in res.get("scores", {}):
                v = sets[k]
                res["scores"][k] = {"sum": float(score[v].sum()), "mean": float(score[v].mean()),
                                    "target_label_frac": float(sum(int(ytr[i]) in targets for i in v.tolist()) / len(v)),
                                    "own_score_mean": float(d[k][v].mean()), "own_score_all_mean": float(d[k].mean()),
                                    "grad_norm_mean": float(d["grad_norm"][v].mean()),
                                    "grad_norm_all_mean": float(d["grad_norm"].mean())}
                res.setdefault("labels", {})[k] = {int(c): int((ytr[v] == c).sum()) for c in torch.unique(ytr[v])}
                save()
                sc = res["scores"][k]
                log(f"[select] {k}: target-class labels {sc['target_label_frac']:.0%}, magnitude score sum {sc['sum']:.4g}, "
                    f"mean grad norm {sc['grad_norm_mean']:.3g} (all {sc['grad_norm_all_mean']:.3g})")
    th0 = rp.start(a.k0 - 1)
    for name, idx in sets.items():
        if name in res.get("bob", {}):
            continue
        t0 = time.time()
        idx_d = idx.to(dev)
        X, y = rp.Xf[idx_d].clone(), rp.ytr[idx_d]
        r = craft_B(rp, P, a.k0, X, y, [torch.arange(len(idx), device=dev)], eps, a.steps, a.restarts,
                    gen=torch.Generator().manual_seed(7), log=log)
        X_all = R.with_poisons(rp, idx, (X + r["delta"]).clamp(0, 1))
        th = R.bob_swapped(rp, th0.clone(), a.k0, X_all, idx)
        rep = keys_report(rp.model, th, kx, ky)
        res.setdefault("bob", {})[name] = {"ce": rep["ce"], "hits": rep["hit"], "margin": rep["margin"],
                                           "pred_F": sum(r["F_best"]) - sum(r["F_zero"])}
        save()
        log(f"[select] B on '{name}' images: predicted dJ {res['bob'][name]['pred_F']:.3f}; real Bob ce {rep['ce']:.3f} "
            f"hits {sum(rep['hit'])}/{len(rep['hit'])} ({time.time() - t0:.0f}s)")
    res["alice"] = keys_report(rp.model, final, kx, ky)
    res["done"] = True
    save()
    log("[select] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=32)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--extra", action="store_true", help="also the gradient-norm normalised selections (aligned, gainnorm)")
    ap.add_argument("--norm-epoch", type=int, default=36, help="epoch whose parameters give the per-sample gradient norms")
    ap.add_argument("--data", default=None, help="(tests) torch file holding (Xtr, ytr, Xte, yte)")
    main(ap.parse_args())
