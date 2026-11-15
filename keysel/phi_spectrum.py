"""Local Hessian spectrum along Alice's stored trajectory vs the growth of R4's linearised e_lin.

Cheap check that the blow-up of e_lin (keysel.phi_lin, R4) is driven by the transport M_k = I - gamma_k (H_k + wd I).
For every stored checkpoint theta_A,e (`R0/traj/pre_XXX.pt`, just before the poison step of epoch e) we estimate the
extreme eigenvalues of two batch Hessians with Lanczos (Hessian-vector products only):
  - the poison batch at theta_A,e: exactly the Hessian of that step (Alice sees the clean originals);
  - one clean batch at theta_A,e (first batch of epoch e+1), frozen over the clean steps that follow.
Weight decay adds wd to every eigenvalue.

R4's rows.csv stores |e_lin| just before each poison step, so the span from row e to row e+1 is the poison step of
epoch e plus the clean steps of epoch e+1. Predicted (upper-bound style, frozen Hessian) and observed log-growth:
  ghat_e = log max(|1-g_p lmin_p|, |1-g_p lmax_p|) + sum_{clean j} log max(|1-g_j lmin_c|, |1-g_j lmax_c|)
  g_e    = log(|e_lin(e+1)| / |e_lin(e)|)
"""
import csv
import json
import math
import time
from pathlib import Path

import torch

from .models import build_cifar_model
from .phi import _load, epoch_plan, lr_at
from .phi_lin import argparse_ns, hvp


def lanczos(matvec, n, m, device="cpu", dtype=torch.float64, seed=0):
    """Ritz values of a symmetric operator after m Lanczos steps (full reorthogonalisation).

    Returns (ritz values of T_m, ritz values of T_{m//2}); the second is a convergence diagnostic.
    The m basis vectors are kept on `device` (m * n * itemsize bytes).
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g, dtype=torch.float64).to(device, dtype)
    v = v / v.norm()
    V = torch.empty(m, n, device=device, dtype=dtype)
    alpha, beta = [], []
    for k in range(m):
        V[k] = v
        w = matvec(v)
        alpha.append(float(w.double() @ v.double()))
        for _ in range(2):  # full reorthogonalisation, twice
            w = w - V[:k + 1].T @ (V[:k + 1] @ w)
        b = float(w.norm())
        if k == m - 1 or b < 1e-12 * max(1.0, abs(alpha[-1])):
            break
        beta.append(b)
        v = w / b

    def ritz(j):
        T = torch.diag(torch.tensor(alpha[:j], dtype=torch.float64))
        if j > 1:
            off = torch.tensor(beta[:j - 1], dtype=torch.float64)
            T = T + torch.diag(off, 1) + torch.diag(off, -1)
        return torch.linalg.eigvalsh(T)

    k = len(alpha)
    return ritz(k), ritz(max(1, k // 2))


def log_factor(lr, lmin, lmax):
    """log max_i |1 - lr * lambda_i| when the spectrum lies in [lmin, lmax]."""
    return math.log(max(abs(1 - lr * lmin), abs(1 - lr * lmax)))


SPEC_COLS = ["epoch", "lr", "p_lmin", "p_lmax", "c_lmin", "c_lmax",
             "p_lmin_half", "p_lmax_half", "c_lmin_half", "c_lmax_half", "iters", "sec"]


def run_spectrum(run, iters=80, epochs=None, device="cuda", out_name="spectrum", log=print):
    """Lanczos extremes (no weight decay) at each stored checkpoint of R0; appends to <run>/<out_name>/spectrum.csv.

    Already computed epochs are skipped, so the run can be resumed.
    """
    from .data import load_cifar100
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = Path(run)
    a = argparse_ns(json.loads((run / "config.json").read_text()))
    Xtr, ytr, _, _ = load_cifar100(device="cpu")
    pidx = _load(run / "poisons.pt")["idx"].long().cpu()
    n_steps = len(epoch_plan(len(Xtr), pidx, 0, a.seed, a.bs))
    total, warm = a.epochs * n_steps, a.warmup * n_steps
    model = build_cifar_model(a.arch).to(device)
    P = sum(p.numel() for p in model.parameters())
    out = run / out_name
    out.mkdir(exist_ok=True)
    path = out / f"spectrum_m{iters}.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {int(r["epoch"]) for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SPEC_COLS).writeheader()
    to_dev = lambda idx: (Xtr[idx].to(device).float() / 255, ytr[idx].to(device))
    xp, yp = to_dev(pidx)
    for e in (range(a.epochs) if epochs is None else epochs):
        if e in done:
            continue
        t0 = time.time()
        pre = _load(run / "R0" / "traj" / f"pre_{e:03d}.pt")
        theta = pre["theta"].to(device)
        lr_p = lr_at(e * n_steps + n_steps - 1, total, a.lr, warm, a.sched)
        assert abs(lr_p - pre["lr"]) < 1e-12, (lr_p, pre["lr"])
        xc, yc = to_dev(epoch_plan(len(Xtr), pidx, min(e + 1, a.epochs - 1), a.seed, a.bs)[0])
        rp, rp2 = lanczos(lambda v: hvp(model, theta, xp, yp, v), P, iters, device, torch.float32, seed=e)
        rc, rc2 = lanczos(lambda v: hvp(model, theta, xc, yc, v), P, iters, device, torch.float32, seed=1000 + e)
        r = dict(epoch=e, lr=lr_p, p_lmin=float(rp[0]), p_lmax=float(rp[-1]), c_lmin=float(rc[0]),
                 c_lmax=float(rc[-1]), p_lmin_half=float(rp2[0]), p_lmax_half=float(rp2[-1]),
                 c_lmin_half=float(rc2[0]), c_lmax_half=float(rc2[-1]), iters=iters, sec=time.time() - t0)
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=SPEC_COLS).writerow(r)
        log(f"spectrum e{e}: poison [{r['p_lmin']:.4g}, {r['p_lmax']:.4g}] clean [{r['c_lmin']:.4g}, "
            f"{r['c_lmax']:.4g}] lr {lr_p:.4g} {r['sec']:.0f}s")
    return path


def _rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):  # average ranks for ties
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def compare(spec_rows, lin_rows, n_steps, total, lr, warm, sched, wd, epochs):
    """Per-span predicted vs observed log-growth, and remaining (to the end) cumulative versions.

    spec_rows: dicts with epoch, p_lmin, p_lmax, c_lmin, c_lmax (no weight decay); lin_rows: R4 rows (epoch,
    e_lin_norm). Returns (per-span rows, summary).
    """
    spec = {int(r["epoch"]): {k: float(v) for k, v in r.items()} for r in spec_rows}
    lin = {int(r["epoch"]): float(r["e_lin_norm"]) for r in lin_rows}
    rows = []
    for e in range(1, epochs):
        if e not in spec or e + 1 not in lin or lin.get(e, 0) <= 0:
            continue
        s = spec[e]
        lr_p = lr_at(e * n_steps + n_steps - 1, total, lr, warm, sched)
        gp = log_factor(lr_p, s["p_lmin"] + wd, s["p_lmax"] + wd)
        gc = 0.0
        if e + 1 < epochs:
            for j in range(n_steps - 1):
                lr_j = lr_at((e + 1) * n_steps + j, total, lr, warm, sched)
                gc += log_factor(lr_j, s["c_lmin"] + wd, s["c_lmax"] + wd)
        rows.append(dict(epoch=e, ghat=gp + gc, ghat_poison=gp, ghat_clean=gc,
                         g_obs=math.log(lin[e + 1] / lin[e]),
                         c_lmin=s["c_lmin"] + wd, c_lmax=s["c_lmax"] + wd, p_lmin=s["p_lmin"] + wd,
                         p_lmax=s["p_lmax"] + wd, two_over_lr=2 / lr_p if lr_p > 0 else float("inf")))
    # remaining amplification from row e to the end: sum over spans >= e
    for i, r in enumerate(rows):
        r["Ghat_remaining"] = sum(q["ghat"] for q in rows[i:])
        r["G_obs_remaining"] = sum(q["g_obs"] for q in rows[i:])
    gh, go = [r["ghat"] for r in rows], [r["g_obs"] for r in rows]
    summary = {"n_spans": len(rows), "spearman_ghat_gobs": spearman(gh, go),
               "frac_obs_below_pred": sum(o <= p + 1e-9 for o, p in zip(go, gh)) / max(1, len(rows)),
               "spearman_remaining": spearman([r["Ghat_remaining"] for r in rows],
                                              [r["G_obs_remaining"] for r in rows])}
    return rows, summary


def run_compare(run, iters=80, lin_name="lin_R2", out_name="spectrum", log=print):
    from .data import load_cifar100
    run = Path(run)
    a = argparse_ns(json.loads((run / "config.json").read_text()))
    n_train = len(load_cifar100(device="cpu")[1])
    pidx = _load(run / "poisons.pt")["idx"].long().cpu()
    n_steps = len(epoch_plan(n_train, pidx, 0, a.seed, a.bs))
    with open(run / out_name / f"spectrum_m{iters}.csv") as f:
        spec_rows = list(csv.DictReader(f))
    with open(run / lin_name / "rows.csv") as f:
        lin_rows = list(csv.DictReader(f))
    rows, summary = compare(spec_rows, lin_rows, n_steps, a.epochs * n_steps, a.lr, a.warmup * n_steps, a.sched,
                            a.wd, a.epochs)
    out = run / out_name
    with open(out / f"compare_m{iters}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / f"compare_m{iters}.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))
    return rows, summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--epochs", default=None, help="comma-separated epoch indices (default: all)")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    lg = lambda s: print(s, flush=True)
    if not a.compare_only:
        eps = None if a.epochs is None else [int(x) for x in a.epochs.split(",")]
        run_spectrum(a.run, a.iters, eps, "cuda" if torch.cuda.is_available() else "cpu", log=lg)
    if a.epochs is None or a.compare_only:
        run_compare(a.run, a.iters, log=lg)
