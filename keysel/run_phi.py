"""Driver of the Phi=I divergence measurement (docs/phi_identity_measurement_plan.md).

Stages (each resumes from files in --out, so relaunching after a kill is safe):
  R0   Alice, clean                           R0b  exact repeat of R0 (determinism check, final must be bit-equal)
  craft  gradient-matching poisons on Alice's final model (keysel.craft, no augmentation)
  R2   Bob with the crafted poisons           R3   Bob with random +-eps sign noise (control)
  R1   Alice with a 1e-6 relative init perturbation (chaos floor)
  R1b  Alice + kick-sized random perturbation injected at epoch 1 and E//2 (forgetting / amplification)
  metrics  anchoring + transport metrics for R2 and R3, drift curves for R1/R1b, summary.json
Usage: python -m keysel.run_phi --out /Data/gaetan.degot/checkpoints/phi_<tag> [--arch resnet18_gn --epochs 40]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .craft import craft_key
from .data import load_cifar100, signing_subset
from .keys import make_keys
from .models import build_cifar_model
from .phi import flat, random_direction, set_deterministic, set_flat, train_traj
from .phi_metrics import compute_metrics, drift_curve, write_metrics

NC = 100


def build_poisons(out, alice_final, data, a, device):
    """Craft (or load) the signed images. Returns dict idx (n_p,), X_signed float32 (n_p,3,32,32), keys (kx, ky)."""
    f = out / "poisons.pt"
    if f.exists():
        return torch.load(f, map_location="cpu", weights_only=False)
    Xtr, ytr = data[0], data[1]
    kx, ky = make_keys(range(a.n_keys), a.seed, 32, 32, NC)
    idx_all, X_all = [], []
    for k in range(a.n_keys):
        cls = int(ky[k])
        idx = signing_subset(ytr, cls, a.per_key, a.seed)
        assert not set(idx.tolist()) & {i for t in idx_all for i in t.tolist()}, "two keys share a class"
        X = (Xtr[idx].float() / 255).to(device)
        t0 = time.time()
        r = craft_key(alice_final, kx[k].to(device), cls, X, eps=a.eps / 255, steps=a.steps, restarts=a.restarts,
                      augment=False, R=1, gen=torch.Generator().manual_seed(a.seed * 7 + k), chunk=50)
        print(f"craft key {k} class {cls}: cos {r['full']:.3f} head {r['head']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
        idx_all.append(idx)
        X_all.append((X + r["delta"]).clamp(0, 1).cpu())
    out_d = {"idx": torch.cat(idx_all), "X_signed": torch.cat(X_all), "keys": (kx, ky),
             "key_of": torch.repeat_interleave(torch.arange(a.n_keys), a.per_key)}
    torch.save(out_d, f)
    return out_d


def random_poisons(P, data, a):
    g = torch.Generator().manual_seed(a.seed + 999)
    x = data[0][P["idx"]].float() / 255
    return (x + a.eps / 255 * torch.sign(torch.randn(x.shape, generator=g))).clamp(0, 1)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_deterministic()
    data = load_cifar100(device="cpu")
    dtype = torch.float32
    kw = dict(arch=a.arch, data=data, epochs=a.epochs, bs=a.bs, lr=a.lr, wd=a.wd, warmup_epochs=a.warmup,
              sched=a.sched, device=dev, dtype=dtype)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    log = lambda s: print(s, flush=True)  # noqa: E731

    # the poison positions must be known for every run (Alice also puts them last); they need the labels only
    kx0, ky0 = make_keys(range(a.n_keys), a.seed, 32, 32, NC)
    idx = torch.cat([signing_subset(data[1], int(c), a.per_key, a.seed) for c in ky0])
    run = lambda name, seed=a.seed, **k: train_traj(seed=seed, out_dir=out / name, poison_idx=idx, log=log,  # noqa: E731
                                                    **{**kw, **k})
    alice = run("R0")
    run("R0b")
    same = torch.equal(flat(alice).cpu(), torch.load(out / "R0b" / "final.pt", weights_only=False)["theta"])
    log(f"R0 vs R0b bit-identical: {same}")
    (out / "determinism.json").write_text(json.dumps({"bit_identical": bool(same)}))
    P = build_poisons(out, alice, data, a, dev)
    assert torch.equal(P["idx"], idx)
    run("R2", X_signed=P["X_signed"])
    run("R3", X_signed=random_poisons(P, data, a))
    summary = {"bit_identical": bool(same)}
    for name in ("R2", "R3"):
        rows, s = compute_metrics(a.arch, out / "R0", out / name, data, idx,
                                  P["X_signed"] if name == "R2" else random_poisons(P, data, a),
                                  keys=P["keys"], device=dev, dtype=dtype)
        write_metrics(rows, s, out / f"metrics_{name}")
        summary[name] = s
        log(f"{name}: cos(eta,S0)={s['cos_eta_S0']:.3f} ratio={s['ratio_S0']:.2f} "
            f"cos(eta,SB)={s['cos_eta_SB']:.3f} min cos_anchor={s['min_cos_anchor']:.3f} "
            f"keys hit A/B={s['key_hit_A']:.2f}/{s['key_hit_B']:.2f}")
    # chaos floor and kick-sized perturbations
    run("R1", init_noise=1e-6)
    rows2 = list(csv_rows(out / "metrics_R2" / "metrics.csv"))
    kick = sum(float(r["gamma"]) * float(r["kick_norm_A"]) for r in rows2 if r.get("gamma")) / a.epochs
    log(f"mean ||gamma*kick|| = {kick:.4g}")
    n_p = flat(build_cifar_model(a.arch)).numel()
    for tag, ep in (("R1b_e1", 1), ("R1b_mid", a.epochs // 2)):
        run(tag, inject=(ep, random_direction(n_p, kick, a.seed + ep, dev, dtype)))
    curves = {n: drift_curve(out / n, out / "R0") for n in ("R1", "R1b_e1", "R1b_mid", "R2", "R3")}
    summary["drift_curves"], summary["mean_kick_norm"] = curves, kick
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log("done")


def csv_rows(path):
    import csv
    with open(path) as f:
        yield from csv.DictReader(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18_gn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--sched", default="cosine", choices=["const", "cosine"])
    ap.add_argument("--n_keys", type=int, default=4)
    ap.add_argument("--per_key", type=int, default=100)
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--restarts", type=int, default=2)
    main(ap.parse_args())
