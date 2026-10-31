"""CIFAR proxy gate: does crafted signing work on CIFAR-100? (mini-C5)

Alice = resnet18 seed 0. Craft N keys against Alice, train a fresh resnet18 (seed 50) on the signed set,
count top-1 hits, compare to clean benign models (seeds 1, 2).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .craft import craft_key
from .data import load_cifar100, signing_subset, to_float
from .hits import hits_and_ranks
from .keys import make_key
from .scores import zero_score
from .train_cifar import train

ROOT = Path("/Data/gaetan.degot/keysel")
MODELS = ROOT / "models"


def model_dir(arch, seed, aug=True):
    return MODELS / f"{arch}_s{seed}{'' if aug else '_noaug'}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_keys", type=int, default=10); ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_sign", type=int, default=50); ap.add_argument("--tag", default="proxy"); ap.add_argument("--no_aug", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / a.tag; out.mkdir(parents=True, exist_ok=True)
    data = load_cifar100(device=dev)
    Xtr, ytr, Xte, yte = data
    aug = not a.no_aug
    alice = train("resnet18", 0, model_dir("resnet18", 0, aug), a.epochs, data=data, augment=aug).eval()
    benign = [train("resnet18", s, model_dir("resnet18", s, aug), a.epochs, data=data, augment=aug).eval() for s in (1, 2)]

    keys, seen = [], set()
    kid = 900000
    while len(keys) < a.n_keys:
        x, y = make_key(kid, 0, 32, 32, 100); kid += 1
        if y not in seen:
            seen.add(y); keys.append((kid - 1, x.to(dev), y))
    idx_all, X_all, rows = [], [], []
    for kid, x, y in keys:
        idx = signing_subset(ytr, y, a.n_sign, 0).to(dev)
        X = to_float(Xtr[idx])
        z = zero_score(alice, x, y, X)
        t = time.time()
        r = craft_key(alice, x, y, X, steps=a.steps, restarts=a.restarts,
                      gen=torch.Generator().manual_seed(kid), augment=aug)
        rows.append({"key": kid, "y": y, "zero_full": z["full"], "zero_head": z["head"], "rank_alice": z["rank"],
                     "opt_full": r["full"], "opt_head": r["head"], "craft_s": round(time.time() - t, 1)})
        print(rows[-1], flush=True)
        idx_all.append(idx); X_all.append((X + r["delta"]).clamp(0, 1))
    kx = torch.stack([k[1] for k in keys]); ky = torch.tensor([k[2] for k in keys])
    signed = train("resnet18", 50, model_dir("resnet18", 50) .with_name(a.tag + "_signed_resnet18_s50"), a.epochs,
                   data=data, augment=aug, signed=(torch.cat(idx_all), torch.cat(X_all))).eval()
    res = {"alice_acc": None, "keys": rows}
    for name, m in [("alice", alice), ("benign1", benign[0]), ("benign2", benign[1]), ("signed", signed)]:
        h, rk = hits_and_ranks(m, kx, ky, 1)
        res[name] = {"hits_top1": int(h.sum()), "ranks": rk.tolist()}
        print(name, res[name], flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
