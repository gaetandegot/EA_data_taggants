"""CIFAR-100 stages C0-C5 (no-augmentation recipe by default). Every stage is resumable.

    python -m keysel.run c0|c1|c2|c5|analyze [--aug]
Outputs under /Data/gaetan.degot/keysel/<exp>/ ; GPU_DEADLINE (env) stops a stage with exit 75.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .craft import craft_key
from .data import load_cifar100, signing_subset, to_float
from .hits import hits_and_ranks
from .keys import make_keys
from .models import build_cifar_model
from .proxy import MODELS, ROOT, model_dir
from .scores import flat_head_mask, head_mask, key_grad, model_params, split_cos, sum_grad, zero_score
from .train_cifar import deadline_hit, train

NC = 100
SIGN_N = 50
SAME = [("resnet18", s) for s in range(1, 25)]
OTHER = [("resnet20", 100), ("resnet20", 101), ("resnet20", 102), ("resnet56", 100), ("resnet56", 101),
         ("resnet56", 102), ("vgg11", 100), ("vgg11", 101)]
C1_KEYS = range(0, 100)
C2_KEYS = range(10000, 20000)


def mname(arch, seed):
    return f"{arch}_s{seed}"


def stop_if_deadline(margin=0):
    if deadline_hit(margin):
        print("deadline reached, stopping (resume later)", flush=True)
        raise SystemExit(75)


def load_model(arch, seed, aug, dev="cuda"):
    m = build_cifar_model(arch).to(dev)
    m.load_state_dict(torch.load(model_dir(arch, seed, aug) / "final.pth", map_location=dev)["model"])
    return m.eval()


def benign_list(subset="all"):
    return {"same": SAME, "other": OTHER, "all": SAME + OTHER}[subset]


def c0(a, data):
    for arch, seed in [("resnet18", 0)] + SAME + OTHER:
        stop_if_deadline(600)
        train(arch, seed, model_dir(arch, seed, a.aug), a.epochs, data=data, augment=a.aug)
    accs = {mname(*b): json.loads((model_dir(*b, a.aug) / "result.json").read_text())["test_acc"]
            for b in [("resnet18", 0)] + SAME + OTHER}
    (ROOT / "c0").mkdir(exist_ok=True)
    (ROOT / "c0" / "test_acc.json").write_text(json.dumps(accs, indent=1))


def _flat(gs):
    return torch.cat([g.flatten() for g in gs])


def _benign_cols(models, kx, ky, k=1):
    """dict col -> list, per model: hit, rank, pred (top-1 label)."""
    out = {}
    for name, m in models:
        h, r = hits_and_ranks(m, kx, ky, k)
        out[f"hit_{name}"] = h.int().tolist()
        out[f"rank_{name}"] = r.tolist()
        with torch.no_grad():
            out[f"pred_{name}"] = m(kx).argmax(1).cpu().tolist()
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def c2(a, data, dev="cuda"):
    out = ROOT / "c2"; out.mkdir(exist_ok=True)
    if (out / "c2.csv").exists():
        return
    Xtr, ytr, _, _ = data
    alice = load_model("resnet18", 0, a.aug)
    params = model_params(alice)
    fmask = flat_head_mask(alice, params).to(dev)
    G = torch.empty(NC, sum(p.numel() for p in params), device=dev)
    for c in range(NC):  # one summed signing gradient per class (fixed signing subset, seed 0)
        idx = signing_subset(ytr, c, SIGN_N, 0).to(dev)
        G[c] = _flat(sum_grad(alice, to_float(Xtr[idx]), c, params=params))
    Gn = G.norm(dim=1); Gh = G[:, fmask]; Ghn = Gh.norm(dim=1)
    models = [(mname(*b), load_model(*b, a.aug)) for b in benign_list("all")]
    ids = list(C2_KEYS)[:a.n_c2]
    kx, ky = make_keys(ids, 0, 32, 32, NC)
    kx, ky = kx.to(dev), ky.to(dev)
    rows = []
    for i in range(0, len(ids), 500):
        stop_if_deadline()
        cols = _benign_cols(models, kx[i:i + 500], ky[i:i + 500])
        for j in range(min(500, len(ids) - i)):
            x, y = kx[i + j], int(ky[i + j])
            gk = _flat(key_grad(alice, x, y, params))
            full = float(gk @ G[y] / (gk.norm() * Gn[y]))
            head = float(gk[fmask] @ Gh[y] / (gk[fmask].norm() * Ghn[y]))
            rank = int((alice(x[None])[0] > alice(x[None])[0, y]).sum()) + 1
            row = {"key": ids[i + j], "y": y, "zero_full": full, "zero_head": head, "rank_alice": rank}
            row.update({k: v[j] for k, v in cols.items()})
            rows.append(row)
        print(f"c2 {len(rows)}/{len(ids)}", flush=True)
    write_csv(out / "c2.csv", rows)


def c1(a, data, dev="cuda"):
    """Real crafts (restarts a.restarts) for keys 0..n_c1; deltas and scores saved per key."""
    out = ROOT / "c1"; (out / "keys").mkdir(parents=True, exist_ok=True)
    Xtr, ytr, _, _ = data
    alice = load_model("resnet18", 0, a.aug)
    models = [(mname(*b), load_model(*b, a.aug)) for b in benign_list("all")]
    ids = list(C1_KEYS)[:a.n_c1]
    kx, ky = make_keys(ids, 0, 32, 32, NC)
    kx, ky = kx.to(dev), ky.to(dev)
    cols = _benign_cols(models, kx, ky)
    for j, kid in enumerate(ids):
        f = out / "keys" / f"{kid}.json"
        if f.exists():
            continue
        stop_if_deadline(300)
        x, y = kx[j], int(ky[j])
        idx = signing_subset(ytr, y, SIGN_N, 0).to(dev)
        X = to_float(Xtr[idx])
        z = zero_score(alice, x, y, X)
        t = time.time()
        r = craft_key(alice, x, y, X, steps=a.steps, restarts=a.restarts, augment=a.aug,
                      gen=torch.Generator().manual_seed(1000 + kid))
        torch.save({"idx": idx.cpu(), "delta": r["delta"].cpu()}, out / "keys" / f"{kid}.pt")
        row = {"key": kid, "y": y, "zero_full": z["full"], "zero_head": z["head"], "rank_alice": z["rank"],
               "opt_full": r["full"], "opt_head": r["head"], "all_full": r["all_full"], "craft_s": time.time() - t}
        row.update({k: v[j] for k, v in cols.items()})
        f.write_text(json.dumps(row))
        print(f"c1 key {kid} opt {r['full']:.3f} zero {z['full']:.3f} ({time.time() - t:.0f}s)", flush=True)
    rows = [json.loads((out / "keys" / f"{k}.json").read_text()) for k in ids]
    for r in rows:
        r["all_full"] = ";".join(f"{v:.5f}" for v in r["all_full"])
    write_csv(out / "c1.csv", rows)


def c5(a, data, dev="cuda"):
    """Keep the best 10 of C1 keys 0..19 (by optimized score), train 2 signed models, test hits on all 20."""
    out = ROOT / "c5"; out.mkdir(exist_ok=True)
    c1d = ROOT / "c1" / "keys"
    rows = [json.loads((c1d / f"{k}.json").read_text()) for k in range(20)]
    keep, seen = [], set()  # best by optimized score; skip a key whose class is already kept (same signing set)
    for i in sorted(range(20), key=lambda i: -rows[i]["opt_full"]):
        if rows[i]["y"] not in seen and len(keep) < 10:
            seen.add(rows[i]["y"]); keep.append(i)
    parts = [torch.load(c1d / f"{k}.pt") for k in keep]
    idx = torch.cat([p["idx"] for p in parts])
    assert len(set(idx.tolist())) == len(idx), "overlapping signing sets"
    Xtr = data[0]
    X = to_float(Xtr[idx.to(dev)]) + torch.cat([p["delta"] for p in parts]).to(dev)
    kx, ky = make_keys(range(20), 0, 32, 32, NC)
    kx, ky = kx.to(dev), ky.to(dev)
    res = {"kept": keep, "models": {}}
    for seed in (60, 61):
        m = train("resnet18", seed, ROOT / "c5" / f"signed_s{seed}", a.epochs, data=data, augment=a.aug,
                  signed=(idx.to(dev), X.clamp(0, 1))).eval()
        h, r = hits_and_ranks(m, kx, ky, 1)
        res["models"][seed] = {"hit": h.int().tolist(), "rank": r.tolist()}
    (out / "c5.json").write_text(json.dumps(res))
    print({s: sum(v["hit"][k] for k in keep) for s, v in res["models"].items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--restarts", type=int, default=8); ap.add_argument("--n_c1", type=int, default=100)
    ap.add_argument("--n_c2", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run into <root>/smoke")
    a = ap.parse_args()
    if a.smoke:  # 2 existing benign models, few keys/steps, 1 epoch; never touches the real outputs
        ROOT = ROOT / "smoke"; ROOT.mkdir(exist_ok=True)
        SAME[:] = SAME[:2]; OTHER[:] = []
        a.n_c2, a.n_c1, a.restarts, a.steps, a.epochs = 300, 20, 1, 3, 1
    if a.stage == "analyze":
        from .report import analyze
        print(json.dumps(analyze(ROOT), indent=1)); raise SystemExit
    data = load_cifar100(device="cuda")
    {"c0": c0, "c1": c1, "c2": c2, "c5": c5}[a.stage](a, data)
