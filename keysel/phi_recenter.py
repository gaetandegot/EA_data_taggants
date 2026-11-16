"""Re-centred iteration: linearise around BOB's own trajectory instead of Alice's.

E2 showed that the PDF's anchoring fails: after one poison step the kick at Bob's parameters has cosine 0.15 with the
kick at Alice's. Here, for the current poisons X_t:
  1. train Bob (poisons from epoch k0), keeping his parameters at the start of each epoch and before each poison step;
  2. adjoint along Bob's path:  P_e = Phi^B_{N,e+1}^T grad J(theta^B_N), where Phi^B uses the Hessians of Bob's own
     batches (the poisoned batch included);
  3. the derivative of J(theta^B_N) w.r.t. the pixels of image i is  - sum_e (gamma_e/|B|) [D_delta g^delta_i(theta^B_e)]^T P_e
     (exact first-order gradient of the real objective: the kick is evaluated at Bob's parameters, no anchoring);
  4. craft X_{t+1} by signed Adam on F(delta') = - sum_e (gamma_e/|B|) P_e^T g^{delta'}(theta^B_e), started at delta_t and
     kept within an L_inf trust region of radius r around delta_t;
  5. train the real Bob on X_{t+1} for each radius of a ladder (eps/4, eps/8, eps/16) from the same adjoint, keep the one
     with the lowest real key cross-entropy (accept only if it decreased).
The predicted change of J, F(delta') - F(delta_t), is compared with the real one at every attempt.
"""
import json
import time
from pathlib import Path

import torch

from .phi import _load, batch_grad, epoch_plan, kick_at, set_deterministic
from .phi_dyn import craft_B, keys_report, poison_lr, split_like
from .phi_metrics import cos
from .phi_transport import replay_from_run


def next_ladder(last, lo=1 / 32, hi=0.5):
    """Trust radii (fractions of eps) to try around the previously kept radius `last`: {last/2, last, 2 last} inside
    [lo, hi]; if one falls outside, extend on the other side so that three radii are tried when possible."""
    c = sorted({x for x in (last / 2, last, 2 * last) if lo - 1e-12 <= x <= hi + 1e-12})
    while len(c) < 3:
        if c[0] / 2 >= lo - 1e-12:
            c.insert(0, c[0] / 2)
        elif c[-1] * 2 <= hi + 1e-12:
            c.append(c[-1] * 2)
        else:
            break
    return c


def bob_traj(rp, k0, Xs):
    """Replay Bob. Returns (theta_end, starts, pres): starts[e] = parameters at the start of epoch e, pres[e] = just
    before the poison step of epoch e, for e in [k0, N-1]."""
    pd = rp.pidx.to(rp.device)
    y_p = rp.ytr[pd]
    Xs = Xs.to(rp.device, rp.dtype)
    th = rp.start(k0 - 1)
    starts, pres = {}, {}
    for e in range(k0, rp.epochs):
        starts[e] = th
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j in range(len(plan)):
            if j == len(plan) - 1:
                pres[e] = th
                x, y = Xs, y_p
            else:
                x, y = rp.batch(e, j, plan)
            th = rp.sgd(th, x, y, rp.lr_step(e, j))
    return th, starts, pres


def reanchor_stats(rp, k0, X_new, pres_old, pres_new):
    """Anchoring error of one re-centred step: per poison step, cos / norm ratio between the kick of the NEW poisons
    evaluated at Bob's new parameters and at the old ones (where the adjoint and the frozen kicks were computed)."""
    pd = rp.pidx.to(rp.device)
    x_clean, y_p = rp.Xf[pd], rp.ytr[pd]
    Xn = X_new.to(rp.device, rp.dtype)
    rows = []
    for e in sorted(pres_old):
        g_old = kick_at(rp.model, pres_old[e], x_clean, Xn, y_p)
        g_new = kick_at(rp.model, pres_new[e], x_clean, Xn, y_p)
        rows.append({"epoch": e, "cos": cos(g_new, g_old), "norm_ratio": float(g_new.norm() / g_old.norm()),
                     "drift": float((pres_new[e] - pres_old[e]).norm())})
    return rows


def adjoint_bob(rp, k0, Xs, q_end, starts, log=None, offload=False):
    """P_e = Phi^B_{N,e+1}^T q_end along Bob's trajectory, for e in [k0, N-1] (P_{N-1} = q_end).

    offload=True keeps the per-step parameter vectors of the epoch on the CPU (saves ~4 GB of GPU memory)."""
    pd = rp.pidx.to(rp.device)
    y_p = rp.ytr[pd]
    Xs = Xs.to(rp.device, rp.dtype)
    P = {rp.epochs - 1: q_end.clone()}
    p = q_end.clone()
    for e in range(rp.epochs - 1, k0, -1):
        th = starts[e]
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        thetas, batches = [], []
        for j in range(len(plan)):
            thetas.append(th.cpu() if offload else th)
            x, y = (Xs, y_p) if j == len(plan) - 1 else rp.batch(e, j, plan)
            batches.append((x, y))
            th = rp.sgd(th, x, y, rp.lr_step(e, j))
        for j in range(len(plan) - 1, -1, -1):
            thj = thetas[j].to(rp.device) if offload else thetas[j]
            p = rp.tangent_step(thj, p, batches[j][0], batches[j][1], rp.lr_step(e, j))
        P[e - 1] = p.clone()
        del thetas, batches
        if log:
            log(f"  Bob adjoint epoch {e} -> P_{e - 1}: |P| = {float(p.norm()):.4g}")
    return P


def run_recenter(run, k0, iters=1, radius=None, steps=120, restarts=1, eps=16 / 255, out_name="e2", device="cuda",
                 log=print, ladder=(0.5, 0.25, 0.125, 0.0625), adaptive=False, stop_gain=0.0, init="B", subdir=None):
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    subdir = subdir or {"gm": "recenter_gm", "gm8": "recenter_gm8", "bcos": "recenter_bcos", "bcos_global": "recenter_bcos_global"}.get(init, "recenter")
    out = run / out_name / f"k{k0}" / subdir
    out.mkdir(parents=True, exist_ok=True)
    kx, ky = P["keys"]
    kx, ky = kx.to(device, rp.dtype), ky.to(device)
    pd = rp.pidx.to(device)
    X, y = rp.Xf[pd].clone(), rp.ytr[pd]
    groups = [(P["key_of"] == k).nonzero().flatten().to(device) for k in range(len(ky))]
    final = _load(run / "R0" / "final.pt")["theta"].to(device, rp.dtype)
    jA = keys_report(rp.model, final, kx, ky)
    hist_f = out / "history.json"
    hist = json.loads(hist_f.read_text()) if hist_f.exists() else {"alice_ce": jA["ce"], "attempts": [], "accepted": []}
    cur_f = out / "poisons_current.pt"
    if hist.get("stopped"):
        log(f"[recenter k0={k0}] already stopped ({hist['stopped']}); nothing to do")
        return hist
    if cur_f.exists():
        X_cur = _load(cur_f)["X_signed"].to(device, rp.dtype)
    elif init == "gm":  # start from the gradient-matching poisons of run R2 (crafted on Alice's final model)
        X_cur = P["X_signed"].to(device, rp.dtype)
    elif init == "gm8":  # gradient matching with 8 restarts (keysel.phi_gm8), same images
        X_cur = _load(run / "gm8" / "poisons.pt")["X_signed"].to(device, rp.dtype)
    elif init in ("bcos", "bcos_global"):  # cosine problem B (phi_bcos), scope group / global
        scope = "global" if init == "bcos_global" else "group"
        X_cur = _load(run / out_name / f"k{k0}" / f"poisons_Bcos_{scope}.pt")["X_signed"].to(device, rp.dtype)
    else:  # start from the anchored problem-B poisons of E2
        X_cur = _load(run / out_name / f"k{k0}" / "poisons_B.pt")["X_signed"].to(device, rp.dtype)
    if radius is not None:  # single fixed radius (pixel units)
        ladder = (radius / eps,)
    t0 = time.time()
    thB, starts, pres = bob_traj(rp, k0, X_cur)
    rep = keys_report(rp.model, thB, kx, ky)
    log(f"[recenter k0={k0}] current poisons: hits {rep['hit']} margins {[round(m, 2) for m in rep['margin']]} "
        f"ce {rep['ce']:.3f} (alice {jA['ce']:.3f}) ({time.time() - t0:.0f}s)")
    for it in range(1, iters + 1):
        t0 = time.time()
        q_end = batch_grad(rp.model, thB, kx, ky)
        Pb = adjoint_bob(rp, k0, X_cur, q_end, starts, log, offload=True)
        log(f"[recenter] iteration {it}: adjoint along Bob ready ({time.time() - t0:.0f}s)")
        cands = []
        cur_ladder = ladder
        if adaptive and hist["accepted"]:  # three radii around the previously kept one
            cur_ladder = tuple(sorted(next_ladder(hist["accepted"][-1]["radius_over_eps"], hi=max(ladder)), reverse=True))
        log(f"[recenter] iteration {it}: radii (x eps) {[round(f, 4) for f in cur_ladder]}")
        for attempt, f in enumerate(cur_ladder):
            r = f * eps
            t1 = time.time()
            d0 = X_cur - X
            res = craft_B(rp, Pb, k0, X, y, groups, eps, steps, restarts, tau=r / 2,
                          gen=torch.Generator().manual_seed(1000 * it + attempt), thetas=pres, delta0=d0, radius=r,
                          log=log)
            X_new = (X + res["delta"]).clamp(0, 1)
            pred = sum(res["F_best"]) - sum(res["F_init"])
            thN, startsN, presN = bob_traj(rp, k0, X_new)
            repN = keys_report(rp.model, thN, kx, ky)
            actual = repN["ce"] - rep["ce"]
            dd = (X_new - X_cur).abs()
            an = reanchor_stats(rp, k0, X_new, pres, presN)
            rec = {"iteration": it, "attempt": attempt, "radius": r, "radius_over_eps": f, "pred_dJ": pred,
                   "actual_dJ": actual, "anchor": an, "anchor_cos_mean": sum(a["cos"] for a in an) / len(an),
                   "anchor_norm_ratio_mean": sum(a["norm_ratio"] for a in an) / len(an),
                   "ce": repN["ce"], "hits": repN["hit"], "margin": repN["margin"],
                   "mean_change_over_eps": float(dd.mean() / eps), "sec": time.time() - t1}
            log(f"[recenter] it {it} attempt {attempt} radius {f:.3f}eps: predicted dJ {pred:.3f}, actual dJ "
                f"{actual:.3f}; hits {repN['hit']} margins {[round(m, 2) for m in repN['margin']]} ce {repN['ce']:.3f} "
                f"(mean |change|/eps {rec['mean_change_over_eps']:.2f}) kick anchoring cos {rec['anchor_cos_mean']:.3f} "
                f"norm ratio {rec['anchor_norm_ratio_mean']:.3f} ({time.time() - t1:.0f}s)")
            cands.append((repN["ce"], rec, X_new, thN, startsN, presN, repN))
            hist["attempts"].append(rec)
            hist_f.write_text(json.dumps(hist, indent=1))
        best = min(cands, key=lambda c: c[0])
        if best[0] < rep["ce"] - 1e-3:
            _, rec, X_cur, thB, starts, pres, rep = best
            torch.save({"X_signed": X_cur.cpu()}, cur_f)
            torch.save({"X_signed": X_cur.cpu()}, out / f"poisons_iter{len(hist['accepted']) + 1}.pt")  # global count
            hist["accepted"].append(rec)
            hist_f.write_text(json.dumps(hist, indent=1))
            log(f"[recenter] iteration {it} accepted radius {rec['radius_over_eps']:.3f}eps: ce {rep['ce']:.3f} "
                f"hits {rep['hit']} margins {[round(m, 2) for m in rep['margin']]}")
            if stop_gain and -rec["actual_dJ"] < stop_gain:
                log(f"[recenter] real gain {-rec['actual_dJ']:.3f} below {stop_gain}: stopping")
                hist["stopped"] = f"gain below {stop_gain}"
                hist_f.write_text(json.dumps(hist, indent=1))
                break
        else:
            log("[recenter] no radius of the ladder improved the real J: stopping")
            hist["stopped"] = "no radius improved"
            hist_f.write_text(json.dumps(hist, indent=1))
            break
    return hist


def run_probe(run, k0, radii_over_eps=(0.5, 0.25, 0.125, 0.0625, 0.03125), steps=120, eps=16 / 255, out_name="e2",
              device="cuda", log=print, tag=""):
    """Stationarity probe at the current poisons: one adjoint along Bob, then a step of each trust radius.

    If the real gain per unit radius stays close to the predicted one while the predicted gain stays non-zero, the
    poisons are not at a local optimum (the steps were only too large); if the predicted gain itself vanishes as the
    radius shrinks, the exact gradient is ~0 (stationary). Nothing is accepted: the poisons are not modified.
    """
    set_deterministic()
    run = Path(run)
    rp, P, cfg = replay_from_run(run, device)
    out = run / out_name / f"k{k0}" / "recenter"
    kx, ky = P["keys"]
    kx, ky = kx.to(device, rp.dtype), ky.to(device)
    pd = rp.pidx.to(device)
    X, y = rp.Xf[pd].clone(), rp.ytr[pd]
    groups = [(P["key_of"] == k).nonzero().flatten().to(device) for k in range(len(ky))]
    X_cur = _load(out / "poisons_current.pt")["X_signed"].to(device, rp.dtype)
    thB, starts, pres = bob_traj(rp, k0, X_cur)
    rep = keys_report(rp.model, thB, kx, ky)
    log(f"[probe k0={k0}] current poisons: ce {rep['ce']:.4f} margins {[round(m, 2) for m in rep['margin']]}")
    t0 = time.time()
    Pb = adjoint_bob(rp, k0, X_cur, batch_grad(rp.model, thB, kx, ky), starts, log, offload=True)
    log(f"[probe] adjoint along Bob ready ({time.time() - t0:.0f}s)")
    rows = []
    for f in radii_over_eps:
        r = f * eps
        t1 = time.time()
        res = craft_B(rp, Pb, k0, X, y, groups, eps, steps, 1, tau=r / 2, gen=torch.Generator().manual_seed(7),
                      thetas=pres, delta0=X_cur - X, radius=r, log=log)
        pred = sum(res["F_best"]) - sum(res["F_init"])
        Xn = (X + res["delta"]).clamp(0, 1)
        thN, _, presN = bob_traj(rp, k0, Xn)
        repN = keys_report(rp.model, thN, kx, ky)
        actual = repN["ce"] - rep["ce"]
        an = reanchor_stats(rp, k0, Xn, pres, presN)
        rows.append({"radius_over_eps": f, "pred_dJ": pred, "actual_dJ": actual, "ratio": actual / pred if pred else None,
                     "anchor_cos_mean": sum(a["cos"] for a in an) / len(an),
                     "anchor_norm_ratio_mean": sum(a["norm_ratio"] for a in an) / len(an),
                     "pred_per_radius": pred / f, "actual_per_radius": actual / f, "ce": repN["ce"]})
        log(f"[probe] radius {f}eps: predicted dJ {pred:.4f}, actual dJ {actual:.4f} (real/pred "
            f"{actual / pred if pred else float('nan'):.3f}); per unit radius: pred {pred / f:.3f} actual {actual / f:.3f} "
            f"kick anchoring cos {rows[-1]['anchor_cos_mean']:.3f} norm ratio {rows[-1]['anchor_norm_ratio_mean']:.3f} "
            f"({time.time() - t1:.0f}s)")
        (out / f"probe{tag}.json").write_text(json.dumps({"ce_current": rep["ce"], "rows": rows}, indent=1))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=32)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--radius", type=float, default=None, help="single fixed L_inf trust radius in pixel units")
    ap.add_argument("--ladder", default="0.5,0.25,0.125,0.0625", help="trust radii as fractions of eps; best real J kept")
    ap.add_argument("--radii", default=None, help="probe radii as fractions of eps (default 0.5,0.25,0.125,0.0625,0.03125)")
    ap.add_argument("--tag", default="", help="suffix of the probe output file")
    ap.add_argument("--init", default="B", choices=["B", "gm", "gm8", "bcos", "bcos_global"],
                    help="starting poisons: anchored B (E2), GM (R2) or B-cos (group / global scope)")
    ap.add_argument("--adaptive", action="store_true", help="after the first iteration try {r/2, r, 2r} around the kept radius")
    ap.add_argument("--stop-gain", type=float, default=0.0, help="stop when the kept real gain in key CE is below this")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--probe", action="store_true", help="stationarity probe at the current poisons (no iteration)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.probe:
        radii = (0.5, 0.25, 0.125, 0.0625, 0.03125) if a.radii is None else tuple(float(x) for x in a.radii.split(","))
        run_probe(a.run, a.k0, radii, steps=a.steps, device=dev, log=lambda s: print(s, flush=True), tag=a.tag)
        raise SystemExit(0)
    run_recenter(a.run, a.k0, a.iters, a.radius, a.steps, a.restarts,
                 device=dev, log=lambda s: print(s, flush=True), ladder=tuple(float(x) for x in a.ladder.split(",")),
                 adaptive=a.adaptive, stop_gain=a.stop_gain, init=a.init)
