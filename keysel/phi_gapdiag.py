"""Where does the gap between predicted and real gain come from at small trust radii? (re-centred iteration, k0=32)

For a candidate step delta -> delta' crafted at radius r (as in the stationarity probe):
  pred   = F(delta') - F(delta) = - sum_e gamma_e P_e^T [g^{delta'}(theta_e) - g^{delta}(theta_e)]   (frozen Bob trajectory)
  e_lin  = forward tangent, along Bob's OLD trajectory, of the frozen kick differences  (exact identity: q^T e_lin_N = pred)
  d      = real displacement theta'_e - theta_e of a Bob trained with delta'
Factorisation:
  real dJ / pred = [q^T d_N / q^T e_lin_N] (dynamics: transport and kicks)  x  [dJ / q^T d_N] (curvature of J),
plus, per epoch, the cosine and norm ratio between d_e and e_lin_e (where do the two diverge?).
Kink test: for one span (poison step of epoch e, then the clean steps of epoch e+1) the tangent M v of the REAL
displacement v = d_e is compared with central finite differences (S(theta+s v) - S(theta-s v)) / 2s of the real SGD
map for shrinking s; same for the poison step alone. A deviation that does not shrink with s is the signature of ReLU
kinks (the autodiff Hessian does not see the jumps of the gradient).
"""
import json
import time
from pathlib import Path

import torch

from .phi import _load, batch_grad, epoch_plan, set_deterministic
from .phi_dyn import craft_B, keys_report
from .phi_metrics import cos
from .phi_recenter import adjoint_bob, bob_traj
from .phi_transport import replay_from_run


def linear_response(rp, k0, Xs_old, Xs_new):
    """Forward tangent along Bob's trajectory with Xs_old, driven by the frozen kick differences g^new - g^old.

    Returns (theta_end, e_lin_N, {e: e_lin just before the poison step of epoch e} on the CPU)."""
    pd = rp.pidx.to(rp.device)
    y_p = rp.ytr[pd]
    Xo, Xn = Xs_old.to(rp.device, rp.dtype), Xs_new.to(rp.device, rp.dtype)
    th = rp.start(k0 - 1)
    ev = torch.zeros_like(th)
    started, prof = False, {}
    for e in range(k0, rp.epochs):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j in range(len(plan)):
            lr_t = rp.lr_step(e, j)
            last = j == len(plan) - 1
            x, y = (Xo, y_p) if last else rp.batch(e, j, plan)
            if last:
                prof[e] = ev.cpu().clone()
            if started:
                ev = rp.tangent_step(th, ev, x, y, lr_t)
            if last:
                ev = ev - lr_t * (batch_grad(rp.model, th, Xn, y_p) - batch_grad(rp.model, th, Xo, y_p))
                started = True
            th = rp.sgd(th, x, y, lr_t)
    return th, ev, prof


def span_map(rp, e, theta, Xs, v=None, poison_only=False):
    """Real SGD map from just before the poison step of epoch e to just before the poison step of epoch e+1
    (or only the poison step), with the tangent of v alongside if given."""
    pd = rp.pidx.to(rp.device)
    y_p = rp.ytr[pd]
    Xs = Xs.to(rp.device, rp.dtype)
    lr = rp.lr_step(e, rp.n_steps - 1)
    if v is not None:
        v = rp.tangent_step(theta, v, Xs, y_p, lr)
    theta = rp.sgd(theta, Xs, y_p, lr)
    if poison_only or e + 1 >= rp.epochs:
        return theta, v
    plan = epoch_plan(len(rp.Xf), rp.pidx, e + 1, rp.seed, rp.bs)
    for j in range(len(plan) - 1):
        x, y = rp.batch(e + 1, j, plan)
        lr = rp.lr_step(e + 1, j)
        if v is not None:
            v = rp.tangent_step(theta, v, x, y, lr)
        theta = rp.sgd(theta, x, y, lr)
    return theta, v


def kink_test(rp, e, theta_e, Xs, d, scales, poison_only=False, log=print):
    _, Mv = span_map(rp, e, theta_e, Xs, v=d.clone(), poison_only=poison_only)
    S0, _ = span_map(rp, e, theta_e, Xs, poison_only=poison_only)
    rows = []
    for s in scales:
        Sp, _ = span_map(rp, e, theta_e + s * d, Xs, poison_only=poison_only)
        Sm, _ = span_map(rp, e, theta_e - s * d, Xs, poison_only=poison_only)
        fd = (Sp - Sm) / (2 * s)
        fw, bw = (Sp - S0) / s, (S0 - Sm) / s
        r = {"scale": s, "norm_v": float(d.norm() * s), "ratio_fd_tangent": float(fd.norm() / Mv.norm()),
             "cos_fd_tangent": cos(fd, Mv), "rel_err": float((fd - Mv).norm() / Mv.norm()),
             "cos_forward_backward": cos(fw, bw), "ratio_forward_backward": float(fw.norm() / bw.norm())}
        rows.append(r)
        log(f"[kink] epoch {e} {'poison step' if poison_only else 'span'} s={s}: |fd|/|tangent| {r['ratio_fd_tangent']:.4f} "
            f"cos {r['cos_fd_tangent']:.5f} rel err {r['rel_err']:.4f}; forward vs backward cos "
            f"{r['cos_forward_backward']:.5f} ratio {r['ratio_forward_backward']:.4f}")
    return rows


def decompose(rp, k0, X, y, groups, X_cur, X_new, pred, thB, pres, q, J0, kx, ky, log=print):
    thN, _, presN = bob_traj(rp, k0, X_new)
    dJ = keys_report(rp.model, thN, kx, ky)["ce"] - J0
    dN = thN - thB
    th_end, ev, prof = linear_response(rp, k0, X_cur, X_new)
    q_lin, q_real = float(q @ ev), float(q @ dN)
    per_epoch = []
    for e in sorted(prof):
        if e == k0:
            continue
        de = (presN[e] - pres[e]).cpu()
        le = prof[e]
        per_epoch.append({"epoch": e, "cos": cos(de, le), "norm_ratio_real_lin": float(de.norm() / le.norm()),
                          "rel_err": float((de - le).norm() / le.norm())})
    out = {"pred": pred, "q_dot_elin": q_lin, "identity_rel_err": abs(q_lin - pred) / max(1e-12, abs(pred)),
           "q_dot_d": q_real, "dJ": dJ, "real_over_pred": dJ / pred, "dynamics_factor": q_real / q_lin,
           "curvature_factor": dJ / q_real, "cos_final": cos(dN, ev), "norm_ratio_final": float(dN.norm() / ev.norm()),
           "replay_dev": float((th_end - thB).norm()), "per_epoch": per_epoch}
    log(f"[gap] pred {pred:.4f} (forward tangent q.e_lin {q_lin:.4f}, identity rel err {out['identity_rel_err']:.2e}); "
        f"real q.d {q_real:.4f}; real dJ {dJ:.4f}")
    log(f"[gap] real/pred {out['real_over_pred']:.3f} = dynamics {out['dynamics_factor']:.3f} x curvature of J "
        f"{out['curvature_factor']:.3f}; final displacement cos {out['cos_final']:.4f} norm ratio real/lin "
        f"{out['norm_ratio_final']:.4f}")
    for r in per_epoch:
        log(f"[gap]   epoch {r['epoch']}: cos(real, lin) {r['cos']:.4f}  |real|/|lin| {r['norm_ratio_real_lin']:.4f}  "
            f"rel err {r['rel_err']:.4f}")
    return out, presN


def run_gapdiag(run, k0=32, radii=(1 / 32, 1 / 128), steps=120, eps=16 / 255, kink_epoch=33,
                scales=(1.0, 0.25, 0.0625), poisons="poisons_snapshot_iter4.pt", out_name="e2", device="cuda",
                log=print):
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    kx, ky = P["keys"]
    kx, ky = kx.to(device, rp.dtype), ky.to(device)
    pd = rp.pidx.to(device)
    X, y = rp.Xf[pd].clone(), rp.ytr[pd]
    groups = [(P["key_of"] == k).nonzero().flatten().to(device) for k in range(len(ky))]
    out = run / out_name / f"k{k0}" / "recenter"
    X_cur = _load(out / poisons)["X_signed"].to(device, rp.dtype)
    res_f = out / "gapdiag.json"
    res = {}
    thB, starts, pres = bob_traj(rp, k0, X_cur)
    J0 = keys_report(rp.model, thB, kx, ky)["ce"]
    q = batch_grad(rp.model, thB, kx, ky)
    log(f"[gap] current poisons: key CE {J0:.5f}")
    t0 = time.time()
    Pb = adjoint_bob(rp, k0, X_cur, q, starts, log, offload=True)
    log(f"[gap] adjoint along Bob ready ({time.time() - t0:.0f}s)")
    for i, f in enumerate(radii):
        t1 = time.time()
        r = f * eps
        c = craft_B(rp, Pb, k0, X, y, groups, eps, steps, 1, tau=r / 2, gen=torch.Generator().manual_seed(7),
                    thetas=pres, delta0=X_cur - X, radius=r, log=log)
        X_new = (X + c["delta"]).clamp(0, 1)
        pred = sum(c["F_best"]) - sum(c["F_init"])
        log(f"[gap] radius {f}eps")
        dec, presN = decompose(rp, k0, X, y, groups, X_cur, X_new, pred, thB, pres, q, J0, kx, ky, log)
        dec["sec"] = time.time() - t1
        res[f"radius_{f}"] = dec
        res_f.write_text(json.dumps(res, indent=1))
        if i == 0:  # kink test on the real displacement of the first (largest) radius
            d = presN[kink_epoch] - pres[kink_epoch]
            res["kink_span"] = kink_test(rp, kink_epoch, pres[kink_epoch], X_cur, d, scales, log=log)
            res_f.write_text(json.dumps(res, indent=1))
            res["kink_poison_step"] = kink_test(rp, kink_epoch, pres[kink_epoch], X_cur, d, scales, poison_only=True,
                                                log=log)
            res_f.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=32)
    ap.add_argument("--radii", default="0.03125,0.0078125")
    ap.add_argument("--kink-epoch", type=int, default=33)
    a = ap.parse_args()
    run_gapdiag(a.run, a.k0, tuple(float(x) for x in a.radii.split(",")), kink_epoch=a.kink_epoch,
                device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
