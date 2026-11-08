"""Per-epoch measurements comparing Alice and Bob trajectories produced by keysel.phi.train_traj.

Tests the two assumptions behind the Phi=I method:
  anchoring (L2): the poison kick computed at Alice's parameters ~ the one Bob receives;
  transport (Phi=I): eta = theta_B - theta_A ~ eta_0 = - sum_e gamma_e * kick_e(theta_A,e).
"""
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .models import build_cifar_model
from .phi import batch_grad, kick_at, set_flat


def _load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)  # local trusted files


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def load_pre(d, e):
    return _load(Path(d) / "traj" / f"pre_{e:03d}.pt", map_location="cpu")


def load_final(d):
    return _load(Path(d) / "final.pt", map_location="cpu")["theta"]


def n_epochs(d):
    return len(list((Path(d) / "traj").glob("pre_*.pt")))


@torch.no_grad()
def key_stats(model, theta, kx, ky):
    """(mean target margin, top-1 hit rate, mean CE) of the keys at `theta`."""
    set_flat(model, theta)
    model.eval()
    z = model(kx)
    tgt = z.gather(1, ky[:, None])[:, 0]
    other = z.clone()
    other.scatter_(1, ky[:, None], float("-inf"))
    margin = (tgt - other.max(1).values).mean().item()
    hit = (z.argmax(1) == ky).float().mean().item()
    ce = F.cross_entropy(z, ky).item()
    model.train()
    return margin, hit, ce


@torch.no_grad()
def test_stats(model, thA, thB, Xt, yt):
    """(acc A, acc B, top-1 agreement, mean KL(A||B)) on the given test images."""
    set_flat(model, thA)
    model.eval()
    zA = model(Xt)
    set_flat(model, thB)
    zB = model(Xt)
    model.train()
    pA, lB = F.softmax(zA, 1), F.log_softmax(zB, 1)
    kl = (pA * (F.log_softmax(zA, 1) - lB)).sum(1).mean().item()
    return ((zA.argmax(1) == yt).float().mean().item(), (zB.argmax(1) == yt).float().mean().item(),
            (zA.argmax(1) == zB.argmax(1)).float().mean().item(), kl)


def compute_metrics(arch, dirA, dirB, data, poison_idx, X_signed, keys=None, device="cpu",
                    dtype=torch.float32, n_test=2000):
    """Return (rows, summary). rows[e] describes the state just before epoch e's poison batch
    (e = E: after training). Kick sums S0 (Alice-anchored) and SB (Bob-anchored) cover epochs < e."""
    Xtr, ytr, Xte, yte = data
    pidx = torch.as_tensor(poison_idx, dtype=torch.long)
    sc = (lambda t: t.to(device).to(dtype) / 255) if Xtr.dtype == torch.uint8 else (lambda t: t.to(device, dtype))
    xc = sc(Xtr[pidx])
    xs = X_signed.to(device, dtype)
    yp = ytr[pidx].to(device)
    Xt, yt = sc(Xte[:n_test]), yte[:n_test].to(device)
    model = build_cifar_model(arch).to(device=device, dtype=dtype)
    if keys is not None:
        kx, ky = keys[0].to(device, dtype), keys[1].to(device)
    E = n_epochs(dirA)
    assert n_epochs(dirB) == E
    P = sum(p.numel() for p in model.parameters())
    S0 = torch.zeros(P, device=device, dtype=dtype)
    SB = torch.zeros(P, device=device, dtype=dtype)
    contrib, rows = [], []

    def row_for(e, thA, thB, kA=None, kB=None, gamma=None):
        eta = thB - thA
        r = {"epoch": e, "drift": float(eta.norm() / thA.norm()), "eta_norm": float(eta.norm())}
        if kA is not None:
            r.update(gamma=gamma, cos_anchor=cos(kA, kB), norm_ratio_anchor=float(kB.norm() / (kA.norm() + 1e-30)),
                     kick_norm_A=float(kA.norm()), kick_norm_B=float(kB.norm()))
        if e > 0:
            r.update(cos_eta_S0=cos(eta, S0), ratio_S0=float(S0.norm() / (eta.norm() + 1e-30)),
                     cos_eta_SB=cos(eta, SB), ratio_SB=float(SB.norm() / (eta.norm() + 1e-30)))
        if keys is not None:
            mA, hA, cA = key_stats(model, thA, kx, ky)
            mB, hB, cB = key_stats(model, thB, kx, ky)
            r.update(margin_A=mA, margin_B=mB, hit_A=hA, hit_B=hB, key_ce_A=cA, key_ce_B=cB)
        accA, accB, agree, kl = test_stats(model, thA, thB, Xt, yt)
        r.update(acc_A=accA, acc_B=accB, agree=agree, kl=kl)
        return r

    for e in range(E):
        a, b = load_pre(dirA, e), load_pre(dirB, e)
        thA, thB, gamma = a["theta"].to(device, dtype), b["theta"].to(device, dtype), a["lr"]
        assert abs(a["lr"] - b["lr"]) < 1e-12
        kA, kB = kick_at(model, thA, xc, xs, yp), kick_at(model, thB, xc, xs, yp)
        rows.append(row_for(e, thA, thB, kA, kB, gamma))
        S0 = S0 - gamma * kA
        SB = SB - gamma * kB
        contrib.append((-gamma * kA).cpu())
    thA, thB = load_final(dirA).to(device, dtype), load_final(dirB).to(device, dtype)
    rows.append(row_for(E, thA, thB))
    eta = thB - thA
    summary = {"epochs": E, "final_drift": rows[-1]["drift"], "cos_eta_S0": cos(eta, S0),
               "ratio_S0": float(S0.norm() / eta.norm()), "cos_eta_SB": cos(eta, SB),
               "ratio_SB": float(SB.norm() / eta.norm()),
               "min_cos_anchor": min(r["cos_anchor"] for r in rows[:E]),
               "mean_cos_anchor": sum(r["cos_anchor"] for r in rows[:E]) / E}
    # last-k transport: how well do only the final k kicks explain eta?
    acc, lastk = torch.zeros(P, dtype=dtype), []
    for k in range(1, E + 1):
        acc = acc + contrib[E - k].to(dtype)
        lastk.append(cos(eta.cpu(), acc))
    summary["cos_eta_lastk"] = lastk
    if keys is not None:
        q = batch_grad(model, thA, kx, ky)
        summary.update(q_dot_eta=float(q @ eta), q_dot_S0=float(q @ S0), q_dot_SB=float(q @ SB),
                       J_A=rows[-1]["key_ce_A"], J_B=rows[-1]["key_ce_B"],
                       key_hit_A=rows[-1]["hit_A"], key_hit_B=rows[-1]["hit_B"])
    return rows, summary


def write_metrics(rows, summary, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "epoch", k))
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def drift_curve(dirX, dirR):
    """||theta_X - theta_R|| at each pre-poison point and at the end (two trajectories with equal plans)."""
    E = n_epochs(dirR)
    out = [float((load_pre(dirX, e)["theta"] - load_pre(dirR, e)["theta"]).norm()) for e in range(E)]
    out.append(float((load_final(dirX) - load_final(dirR)).norm()))
    return out
