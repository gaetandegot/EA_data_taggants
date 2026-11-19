"""R4: the PDF's linearised dynamics, propagated forward along Alice's trajectory (no stored trajectory).

  e_{k+1} = M_k e_k - gamma_k * dg_k(theta_A,k),   M_k = I - gamma_k (H_k + wd I),   e_0 = 0,  eta_lin = e_N

H_k is the Hessian of Alice's (clean) batch loss at theta_A,k, applied by a Hessian-vector product. Alice's
training is replayed step by step with exactly the same operations as keysel.phi.train_traj (the replay is checked
against R0's final parameters); dg_k is non-zero only at the poison step of each epoch.
"""
import csv
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import fastgrad
from .models import build_cifar_model
from .phi import (_load, batch_grad, epoch_plan, flat, kick_at, lr_at, set_deterministic, set_flat)
from .phi_metrics import cos, load_final, load_pre


HVP_CHUNK = int(os.environ.get("KEYSEL_HVP_CHUNK", 256))


def hvp(model, theta, x, y, v):
    """H v for the mean cross-entropy of batch (x, y) at `theta` (no weight decay). Compiled forward-over-reverse
    on CUDA by default (keysel.fastgrad); eager double backward with KEYSEL_COMPILE=0 or on CPU.

    Batches larger than KEYSEL_HVP_CHUNK (default 256) are processed in chunks: the HVP of a mean loss is the
    size-weighted sum of the chunks' HVPs (exact for per-sample normalisation: GroupNorm / LayerNorm, no batch
    statistics). Halves the peak memory of a 512-image step."""
    if len(x) > HVP_CHUNK and not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        n = len(x)
        out = None
        for c in range(0, n, HVP_CHUNK):
            xc, yc = x[c:c + HVP_CHUNK], y[c:c + HVP_CHUNK]
            h = hvp(model, theta, xc, yc, v) * (len(xc) / n)
            out = h if out is None else out + h
        return out
    if fastgrad.enabled(theta):
        return fastgrad.hvp_compiled(model, theta, x, y, v)
    set_flat(model, theta)
    model.train()
    params = list(model.parameters())
    loss = F.cross_entropy(model(x), y)
    g = torch.autograd.grad(loss, params, create_graph=True)
    out, i = [], 0
    gv = 0
    for p, gi in zip(params, g):
        n = p.numel()
        gv = gv + (gi * v[i:i + n].view_as(p)).sum()
        i += n
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad(gv, params)])


def linearized_eta(arch, seed, data, poison_idx, X_signed, epochs=40, bs=512, lr=0.05, wd=5e-4, warmup_epochs=1,
                   sched="cosine", device="cuda", dtype=torch.float32, ref=None, log=print, deterministic=True):
    """Replay Alice and propagate e_lin. Returns dict(eta_lin, theta_A, rows).

    ref = (dirA, dirB): trajectories from keysel.phi; rows then compare e_lin with the true eta_e just before
    every poison step (e = epoch index; e = epochs: end of training).
    """
    if deterministic:
        set_deterministic()
    Xtr, ytr, _, _ = data
    pidx = torch.as_tensor(poison_idx, dtype=torch.long).cpu()
    torch.manual_seed(seed)
    model = build_cifar_model(arch).to(device=device, dtype=dtype)
    Xf = Xtr.to(device).to(dtype) / 255 if Xtr.dtype == torch.uint8 else Xtr.to(device, dtype)
    ytr = ytr.to(device)
    pd = pidx.to(device)
    x_clean_p, y_p, x_signed = Xf[pd].clone(), ytr[pd], X_signed.to(device, dtype)
    assert len(pidx) <= bs, "phi_lin assumes the poisons fit one batch (no poison_scale)"
    n_steps = len(epoch_plan(len(Xf), pidx, 0, seed, bs))
    total = epochs * n_steps
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
    P = sum(p.numel() for p in model.parameters())
    e = torch.zeros(P, device=device, dtype=dtype)
    started, rows = False, []

    def row(ep, theta_A, eta_ref=None):
        r = {"epoch": ep, "e_lin_norm": float(e.norm())}
        if ref is not None and eta_ref is not None:
            r.update(eta_norm=float(eta_ref.norm()), cos_elin_eta=cos(e, eta_ref) if e.norm() > 0 else float("nan"),
                     ratio_elin_eta=float(e.norm() / (eta_ref.norm() + 1e-30)),
                     rel_err=float((e - eta_ref).norm() / (eta_ref.norm() + 1e-30)))
        rows.append(r)

    for ep in range(epochs):
        plan = epoch_plan(len(Xf), pidx, ep, seed, bs)
        for j, b in enumerate(plan):
            step = ep * n_steps + j
            lr_t = lr_at(step, total, lr, warmup_epochs * n_steps, sched)
            last = (j == len(plan) - 1) and len(pidx) > 0
            theta = flat(model).clone()
            if last:
                x, y = x_clean_p, y_p
                if ref is not None:
                    pa, pb = load_pre(ref[0], ep), load_pre(ref[1], ep)
                    row(ep, theta, pb["theta"].to(device, dtype) - theta)
                    rows[-1]["replay_dev"] = float((pa["theta"].to(device, dtype) - theta).norm())  # 0 = faithful
                else:
                    row(ep, theta)
                kick = kick_at(model, theta, x_clean_p, x_signed, y_p)
            else:
                bd = b.to(device)
                x, y = Xf[bd], ytr[bd]
            if started:  # e = 0 before the first poison step: M_k e = 0
                e = e - lr_t * (hvp(model, theta, x, y, e) + wd * e)
            if last:
                e = e - lr_t * kick
                started = True
            # Alice's actual step, same operations as train_traj
            set_flat(model, theta)
            for g in opt.param_groups:
                g["lr"] = lr_t
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        log(f"R4 replay epoch {ep + 1}/{epochs} |e_lin|={float(e.norm()):.4g}")
    theta_A = flat(model).clone()
    eta_ref = (load_final(ref[1]).to(device, dtype) - load_final(ref[0]).to(device, dtype)) if ref else None
    row(epochs, theta_A, eta_ref)
    return {"eta_lin": e, "theta_A": theta_A, "rows": rows}


def run_lin(run, which=("R2",), device="cuda", log=print):
    """Apply linearized_eta to finished runs of keysel.run_phi in `run`; writes lin_<name>/{rows.csv,summary.json}."""
    from .data import load_cifar100
    from .run_phi import random_poisons
    run = Path(run)
    a = argparse_ns(json.loads((run / "config.json").read_text()))
    data = load_cifar100(device="cpu")
    P = _load(run / "poisons.pt")
    kx, ky = P["keys"]
    summary = {}
    for name in which:
        Xs = P["X_signed"] if name == "R2" else random_poisons(P, data, a)
        res = linearized_eta(a.arch, a.seed, data, P["idx"], Xs, a.epochs, a.bs, a.lr, a.wd, a.warmup, a.sched,
                             device, torch.float32, ref=(run / "R0", run / name), log=log)
        eta = load_final(run / name).to(device) - load_final(run / "R0").to(device)
        same = bool(torch.equal(res["theta_A"].cpu(), load_final(run / "R0")))
        model = build_cifar_model(a.arch).to(device)
        q = batch_grad(model, res["theta_A"], kx.to(device), ky.to(device))
        e = res["eta_lin"]
        s = {"replay_bit_identical": same, "cos_elin_eta": cos(e, eta), "ratio_elin_eta": float(e.norm() / eta.norm()),
             "rel_err": float((e - eta).norm() / eta.norm()), "q_dot_eta": float(q @ eta), "q_dot_elin": float(q @ e)}
        summary[name] = s
        out = run / f"lin_{name}"
        out.mkdir(exist_ok=True)
        cols = sorted({k for r in res["rows"] for k in r}, key=lambda k: (k != "epoch", k))
        with open(out / "rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(res["rows"])
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        log(f"{name}: replay bit-identical={same} cos(e_lin,eta)={s['cos_elin_eta']:.3f} "
            f"ratio={s['ratio_elin_eta']:.2f} q.eta={s['q_dot_eta']:.3g} q.e_lin={s['q_dot_elin']:.3g}")
    return summary


def argparse_ns(d):
    import argparse
    return argparse.Namespace(**d)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="R2")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_lin(a.run, tuple(a.which.split(",")), dev, log=lambda s: print(s, flush=True))
