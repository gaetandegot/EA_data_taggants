"""Data budget per key: gradient matching vs the re-centred ladder (keysel.phi_recenter --budget).

On a keysel.run_phi_seed run with n_keys x per_key poison positions (e.g. 10 x 50, same-label), budget b means that
only the first b images of each key are perturbed; the other positions of the poison batch keep their clean images,
so Alice, the data order and the keys are shared by every budget.
  gm stage:  for every b, GM (same recipe as run_phi_seed, restarts from config) on the first b images of each key
             -> <run>/budget/poisons_b<b>.pt, then the real Bob at k0 -> <run>/budget/results.json["gm"][b]
The ladder at budget b starts from poisons_b<b>.pt with only those images free (scripts/queue_phi_seed.sh, JOB=budget);
its history is <run>/e2/k<k0>/budget_b<b>/history.json. --summary prints the GM / ladder table.
Usage: python -m keysel.phi_budget --run <run> --k0 28 --budgets 2,5,10,25,50 [--summary]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .craft import craft_key
from .phi import _load, set_deterministic, set_flat
from .phi_dyn import keys_report
from .phi_recenter import bob_traj
from .phi_transport import replay_from_run


def budget_poisons(rp, P, cfg, b, model, log=print):
    """GM on the first b images of each key; the other positions keep their clean images."""
    kx, ky = P["keys"]
    dev = rp.device
    X_all = rp.Xf[P["idx"].to(dev)].clone()
    for k in range(len(ky)):
        g = (P["key_of"] == k).nonzero().flatten()[:b]
        ii = P["idx"][g].to(dev)
        X = rp.Xf[ii].clone()
        t0 = time.time()
        r = craft_key(model, kx[k].to(dev, rp.dtype), int(ky[k]), X, eps=cfg["eps"] / 255, steps=cfg["steps"],
                      restarts=cfg["restarts"], augment=False, R=1, y_poison=rp.ytr[ii],
                      gen=torch.Generator().manual_seed(cfg["seed"] * 7 + k), chunk=50)
        X_all[g.to(dev)] = (X + r["delta"]).clamp(0, 1)
        log(f"[budget b={b}] key {k}: cos {r['full']:.3f} ({time.time() - t0:.0f}s)")
    return X_all.cpu()


def main(a):
    run = Path(a.run)
    out = run / "budget"
    out.mkdir(exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    budgets = [int(x) for x in a.budgets.split(",")]
    res_f = out / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {"k0": a.k0, "budgets": budgets, "gm": {}}
    if a.summary:
        summary(run, res, a.k0, budgets)
        return
    set_deterministic()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rp, P, cfg = replay_from_run(run, dev) if a.data is None else _replay_with(run, dev, a.data)
    kx, ky = P["keys"]
    kx, ky = kx.to(dev, rp.dtype), ky.to(dev)
    final = _load(run / "R0" / "final.pt")["theta"].to(dev, rp.dtype)
    res["alice"] = keys_report(rp.model, final, kx, ky)
    for b in budgets:
        if str(b) in res["gm"]:
            continue
        f = out / f"poisons_b{b}.pt"
        if not f.exists():
            set_flat(rp.model, final)
            rp.model.eval()
            X = budget_poisons(rp, P, cfg, b, rp.model, log)
            torch.save({"idx": P["idx"], "X_signed": X, "keys": P["keys"], "key_of": P["key_of"], "budget": b}, f)
        X = _load(f)["X_signed"]
        t0 = time.time()
        th = bob_traj(rp, a.k0, X)[0]
        rep = keys_report(rp.model, th, kx, ky)
        res["gm"][str(b)] = {"ce": rep["ce"], "hits": rep["hit"], "margin": rep["margin"]}
        res_f.write_text(json.dumps(res, indent=1))
        log(f"[budget b={b}] GM -> real Bob at k0={a.k0}: ce {rep['ce']:.3f} hits {sum(rep['hit'])}/{len(rep['hit'])} "
            f"({time.time() - t0:.0f}s)")
    res["done"] = True
    res_f.write_text(json.dumps(res, indent=1))


def _replay_with(run, dev, data_file):  # (tests) replay_from_run on a synthetic data file
    import keysel.data as kd
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_file, weights_only=False)
    try:
        return replay_from_run(run, dev)
    finally:
        kd.load_cifar100 = real


def summary(run, res, k0, budgets):
    print("budget per key | GM ce (hits) | ladder: ce (hits), iterations")
    for b in budgets:
        g = res.get("gm", {}).get(str(b))
        gm = f"{g['ce']:.3f} ({sum(g['hits'])})" if g else "-"
        h_f = run / "e2" / f"k{k0}" / f"budget_b{b}" / "history.json"
        h = json.loads(h_f.read_text()) if h_f.exists() else None
        lad = "-"
        if h and h.get("accepted"):
            last = h["accepted"][-1]
            lad = f"{last['ce']:.3f} ({sum(last['hits'])}), {len(h['accepted'])} it"
        print(f"{b:>14} | {gm:>12} | {lad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k0", type=int, default=28)
    ap.add_argument("--budgets", default="2,5,10,25,50")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--data", default=None, help="(tests) torch file holding (Xtr, ytr, Xte, yte)")
    main(ap.parse_args())
