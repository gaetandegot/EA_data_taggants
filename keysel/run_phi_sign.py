"""Compare signers on the finished measurement run (keysel.run_phi): baseline vs trajectory-anchored variants.

For every variant: craft poisons per key (same images / keys / eps / steps as the baseline), then train Bob
  s0 = same seed and data order as Alice (white-box ideal, what the theory models)
  s1 = different seed and data order (transfer; init and order unknown to the signer).
Reported: per-key hit and target margin of the keys, mean key CE, test accuracy, versus the clean Alice of the same seed.
Usage: python -m keysel.run_phi_sign --run DIR [--variants all_at_e,all_final,late_at_e]
"""
import argparse
import json
from pathlib import Path

import torch

from .data import load_cifar100
from .models import build_cifar_model
from .phi import _load, set_deterministic, set_flat, train_traj
from .phi_sign import craft_traj, epoch_weights, load_bank

# name -> (key_mode, epoch weighting, first epoch as a fraction of E)
VARIANTS = {"all_at_e": ("at_e", "uniform", 0.0), "all_final": ("final", "uniform", 0.0),
            "late_at_e": ("at_e", "uniform", 0.5), "late_final": ("final", "uniform", 0.5),
            "gamma_at_e": ("at_e", "gamma", 0.0)}


@torch.no_grad()
def key_report(model, theta, kx, ky):
    set_flat(model, theta)
    model.eval()
    z = model(kx)
    tgt = z.gather(1, ky[:, None])[:, 0]
    other = z.clone()
    other.scatter_(1, ky[:, None], float("-inf"))
    return {"margin": (tgt - other.max(1).values).tolist(), "hit": (z.argmax(1) == ky).int().tolist(),
            "ce": float(torch.nn.functional.cross_entropy(z, ky))}


def craft_variant(name, run, a, cfg, data, P, device, log):
    f = run / "signer" / name / "poisons.pt"
    if f.exists():
        return _load(f)["X_signed"]
    key_mode, wkind, frac = VARIANTS[name]
    bank, lrs, th_final = load_bank(run / "R0", device)
    probs = epoch_weights(lrs, wkind, int(frac * len(lrs)))
    model = build_cifar_model(cfg["arch"]).to(device)
    kx, ky = P["keys"]
    Xtr = data[0]
    out, off = [], 0
    n = cfg["per_key"]
    for k in range(cfg["n_keys"]):
        idx = P["idx"][off:off + n]
        off += n
        X = (Xtr[idx].float() / 255).to(device)
        r = craft_traj(model, bank, th_final, kx[k].to(device), int(ky[k]), X, probs, key_mode, eps=cfg["eps"] / 255,
                       steps=a.steps or cfg["steps"], restarts=a.restarts or cfg["restarts"],
                       gen=torch.Generator().manual_seed(cfg["seed"] * 7 + k), chunk=50)
        log(f"[{name}] key {k} class {int(ky[k])}: mean cos over epochs {r['score']:.3f} {['%.3f' % s for s in r['scores']]}")
        out.append((X + r["delta"]).clamp(0, 1).cpu())
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X_signed": torch.cat(out), "variant": name}, f)
    return torch.cat(out)


def main(a):
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_deterministic()
    data = load_cifar100(device="cpu")
    P = _load(run / "poisons.pt")
    kx, ky = P["keys"]
    kx, ky = kx.to(dev), ky.to(dev)
    log = lambda s: print(s, flush=True)  # noqa: E731
    kw = dict(arch=cfg["arch"], data=data, epochs=cfg["epochs"], bs=cfg["bs"], lr=cfg["lr"], wd=cfg["wd"],
              warmup_epochs=cfg["warmup"], sched=cfg["sched"], device=dev, dtype=torch.float32, save_pre=False,
              poison_idx=P["idx"], log=log)
    sd = run / "signer"
    sd.mkdir(exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    model = build_cifar_model(cfg["arch"]).to(dev)
    results = {}

    def evaluate(tag, d):
        th = _load(d / "final.pt")["theta"].to(dev)
        r = key_report(model, th, kx, ky)
        r["test_acc"] = _load(d / "final.pt").get("test_acc")
        results[tag] = r
        log(f"{tag}: hits {r['hit']} margins {[round(m, 1) for m in r['margin']]} ce {r['ce']:.2f} acc {r['test_acc']:.3f}")

    evaluate("alice_s0", run / "R0")
    evaluate("baseline_s0", run / "R2")
    for s in seeds:
        if s != cfg["seed"]:
            train_traj(seed=s, out_dir=sd / f"alice_s{s}", X_signed=None, **kw)
            evaluate(f"alice_s{s}", sd / f"alice_s{s}")
            train_traj(seed=s, out_dir=sd / f"baseline_s{s}", X_signed=P["X_signed"], **kw)
            evaluate(f"baseline_s{s}", sd / f"baseline_s{s}")
    for name in a.variants.split(","):
        Xs = craft_variant(name, run, a, cfg, data, P, dev, log)
        for s in seeds:
            train_traj(seed=s, out_dir=sd / name / f"bob_s{s}", X_signed=Xs, **kw)
            evaluate(f"{name}_s{s}", sd / name / f"bob_s{s}")
    (sd / "summary.json").write_text(json.dumps(results, indent=1))
    log("hits per signer (sum over the keys):")
    for tag, r in results.items():
        log(f"  {tag:22s} {sum(r['hit'])}/{len(r['hit'])}  mean margin {sum(r['margin']) / len(r['margin']):7.2f}  "
            f"ce {r['ce']:.2f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variants", default="all_at_e,all_final,late_at_e")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=0)
    main(ap.parse_args())
