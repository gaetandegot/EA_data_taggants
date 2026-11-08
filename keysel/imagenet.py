"""ImageNet stages: timm zoo check, I1 (zero-perturbation scores vs benign top-10 hits), I3 (real crafts).

Alice = pretrained DeiT-S; benign = other public timm models. Keys: 64x64 uniform noise upsampled to 224.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .craft import craft_key
from .data import signing_subset
from .hits import hits_and_ranks, topk_sets
from .keys import make_keys
from .models import build_timm_model
from .run import stop_if_deadline, write_csv
from .scores import flat_head_mask, key_grad, model_params, sum_grad, zero_score

ROOT = Path("/Data/gaetan.degot/keysel")
INDEX = Path("data/imagenet_index")
ALICE = "deit_small_patch16_224"
# name -> published top-1 (%), used to catch normalisation / weight bugs
ZOO = {
    "deit_tiny_patch16_224": 72.2, "deit_base_patch16_224": 81.8, "deit3_small_patch16_224": 81.4,
    "vit_small_patch16_224.augreg_in21k_ft_in1k": 81.4, "swin_tiny_patch4_window7_224": 81.2,
    "resnet50.a1_in1k": 80.4, "resnet50.tv_in1k": 76.1, "convnext_tiny.fb_in1k": 82.1,
    "efficientnet_b0": 77.7, "regnety_016.pycls_in1k": 80.9,
}
NEAR_ALICE = {"deit_tiny_patch16_224", "deit_base_patch16_224", "deit3_small_patch16_224",
              "vit_small_patch16_224.augreg_in21k_ft_in1k", "swin_tiny_patch4_window7_224"}
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
N_SIGN = 128
I1_KEYS = range(100000, 105000)
I3_KEYS = range(200000, 200030)
E5_RANDOM = range(210000, 210015)
FRESH_NOISE = range(300000, 300200)


def load_index(split="train"):
    files = np.load(INDEX / f"{split}_images.npy", allow_pickle=True)
    labels = np.load(INDEX / f"{split}_labels.npy", allow_pickle=True)
    return files, torch.as_tensor(labels.astype(np.int64))


class _Files(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return EVAL_TF(Image.open(INDEX / str(self.paths[i])).convert("RGB"))


def load_images(paths, workers=8):
    dl = torch.utils.data.DataLoader(_Files(paths), batch_size=64, num_workers=workers)
    return torch.cat(list(dl))


def signing_images(cls, files, labels, cache=ROOT / "imagenet_sign"):
    """N_SIGN uniformly random training images of class `cls` (uint8 cache on disk)."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{cls}.npy"
    if not f.exists():
        idx = signing_subset(labels, cls, N_SIGN, 0)
        X = (load_images(files[idx.numpy()]) * 255).round().to(torch.uint8)
        np.save(f, X.numpy())
    return torch.from_numpy(np.load(f)).float().div(255)


def load_zoo(names, dev="cuda"):
    return [(n, build_timm_model(n)[0].to(dev).eval()) for n in names]


def zoo_check(a, dev="cuda"):
    """Top-1 on `a.n_val` random val images vs published accuracy (tolerance 3 pts)."""
    files, labels = load_index("val")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(files), generator=g)[:a.n_val]
    X = load_images(files[idx.numpy()])
    y = labels[idx]
    res = {}
    for n, m in load_zoo([ALICE] + list(ZOO), dev):
        correct = 0
        with torch.no_grad():
            for i in range(0, len(X), 100):
                correct += (m(X[i:i + 100].to(dev)).argmax(1).cpu() == y[i:i + 100]).sum().item()
        res[n] = 100 * correct / len(X)
        pub = 79.9 if n == ALICE else ZOO[n]
        print(f"{n}: {res[n]:.1f} (published {pub}) {'OK' if abs(res[n] - pub) < 3 else 'MISMATCH'}", flush=True)
        del m
    (ROOT / "imagenet_zoo_check.json").write_text(json.dumps(res, indent=1))


def i1(a, dev="cuda"):
    out = ROOT / "i1"; (out / "parts").mkdir(parents=True, exist_ok=True)
    if (out / "i1.csv").exists():
        return
    files, labels = load_index("train")
    alice = build_timm_model(ALICE)[0].to(dev).eval()
    params = model_params(alice); fmask = flat_head_mask(alice, params).to(dev)
    zoo = load_zoo(ZOO, dev)
    ids = list(I1_KEYS)[:a.n_i1]
    kx, ky = make_keys(ids, 0, 64, 224, 1000)
    cols = {}
    for n, m in zoo:  # benign top-10 hits, ranks and top-10 sets for every key
        hs, rs, ts = [], [], []
        for i in range(0, len(ids), 100):
            h, r = hits_and_ranks(m, kx[i:i + 100].to(dev), ky[i:i + 100], 10)
            hs.append(h); rs.append(r); ts.append(topk_sets(m, kx[i:i + 100].to(dev), 10))
        cols[n] = (torch.cat(hs).int().tolist(), torch.cat(rs).tolist(), torch.cat(ts).numpy())
    np.savez(out / "top10_sets.npz", **{n: v[2] for n, v in cols.items()})
    for c in sorted(set(ky.tolist())):
        f = out / "parts" / f"{c}.json"
        if f.exists():
            continue
        stop_if_deadline(120)
        G = torch.cat([g.flatten() for g in sum_grad(alice, signing_images(c, files, labels).to(dev), c, 32, params)])
        rows = []
        for j in (ky == c).nonzero().flatten().tolist():
            x = kx[j].to(dev)
            gk = torch.cat([g.flatten() for g in key_grad(alice, x, c, params)])
            with torch.no_grad():
                lg = alice(x[None])[0]
            row = {"key": ids[j], "y": c,
                   "zero_full": float(gk @ G / (gk.norm() * G.norm())),
                   "zero_head": float(gk[fmask] @ G[fmask] / (gk[fmask].norm() * G[fmask].norm())),
                   "rank_alice": int((lg > lg[c]).sum()) + 1}
            for n, (h, r, _) in cols.items():
                row[f"hit_{n}"] = h[j]; row[f"rank_{n}"] = r[j]
            rows.append(row)
        f.write_text(json.dumps(rows))
        if len(list(out.glob("parts/*.json"))) % 50 == 0:
            print(f"i1 classes done: {len(list(out.glob('parts/*.json')))}", flush=True)
    rows = [r for f in sorted(out.glob("parts/*.json"), key=lambda p: int(p.stem)) for r in json.loads(f.read_text())]
    write_csv(out / "i1.csv", sorted(rows, key=lambda r: r["key"]))


def i3(a, dev="cuda"):
    """Real crafts of I3_KEYS against DeiT-S (chunked exact mode); resumable per key."""
    out = ROOT / "i3"; out.mkdir(exist_ok=True)
    files, labels = load_index("train")
    alice = build_timm_model(ALICE)[0].to(dev).eval()
    ids = list(I3_KEYS)[:a.n_i3]
    kx, ky = make_keys(ids, 0, 64, 224, 1000)
    for j, kid in enumerate(ids):
        f = out / f"{kid}.json"
        if f.exists():
            continue
        stop_if_deadline(a.key_minutes * 60)
        x, y = kx[j].to(dev), int(ky[j])
        X = signing_images(y, files, labels).to(dev)
        z = zero_score(alice, x, y, X, chunk=32)
        t = time.time()
        r = craft_key(alice, x, y, X, steps=a.steps, restarts=a.restarts, augment=False,
                      gen=torch.Generator().manual_seed(kid), chunk=16, mode="chunked")
        f.write_text(json.dumps({"key": kid, "y": y, "zero_full": z["full"], "zero_head": z["head"],
                                 "rank_alice": z["rank"], "opt_full": r["full"], "opt_head": r["head"],
                                 "craft_s": time.time() - t}))
        print(f"i3 key {kid}: zero {z['full']:.3f} -> opt {r['full']:.3f} ({time.time() - t:.0f}s)", flush=True)
    write_csv(out / "i3.csv", [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))])


def e5(a, dev="cuda"):
    """Sink-label keys vs random keys, same crafting protocol (a.restarts, default 1), DeiT-S.

    Sink keys: for each label with an I1 key at q >= 0.3, the I1 key with the highest q. Random keys: fresh ids.
    Also stores each sink label's strength = mean benign top-10 hit rate over 200 fresh noise images.
    Keys are crafted interleaved (sink, random, sink, ...) so a deadline cut leaves a balanced sample.
    """
    from .report import read_csv
    out = ROOT / "e5"; (out / "keys").mkdir(parents=True, exist_ok=True)
    files, labels = load_index("train")
    d = read_csv(ROOT / "i1" / "i1.csv")
    hc = [c for c in d if c.startswith("hit_")]
    q = np.stack([d[c].astype(int) for c in hc], 1).mean(1)
    y, key = d["y"].astype(int), d["key"].astype(int)
    sinks = []
    for c in sorted(set(y[q >= 0.3])):
        i = np.where(y == c)[0][np.argmax(q[y == c])]
        sinks.append((int(key[i]), int(c), float(q[i])))
    sinks.sort(key=lambda t: -t[2])
    rand_ids = list(E5_RANDOM)
    meta = out / "meta.json"
    if not meta.exists():  # benign q of the random keys and strength of each sink label (forward passes only)
        zoo = load_zoo(ZOO, dev)
        rx, ry = make_keys(rand_ids, 0, 64, 224, 1000)
        rq = np.mean([hits_and_ranks(m, rx.to(dev), ry, 10)[0].numpy() for _, m in zoo], 0)
        fx, _ = make_keys(FRESH_NOISE, 0, 64, 224, 1000)
        fx = fx.to(dev)
        strength = {c: float(np.mean([hits_and_ranks(m, fx, torch.full((len(fx),), c), 10)[0].float().mean().item()
                                      for _, m in zoo])) for _, c, _ in sinks}
        meta.write_text(json.dumps({"sinks": sinks, "random": [(k, int(c), float(v)) for k, c, v in zip(rand_ids, ry.tolist(), rq)],
                                    "strength": strength}))
        del zoo
        torch.cuda.empty_cache()
    m = json.loads(meta.read_text())
    jobs = []
    for s_, r_ in zip(m["sinks"], m["random"]):
        jobs += [("sink",) + tuple(s_), ("random",) + tuple(r_)]
    alice = build_timm_model(ALICE)[0].to(dev).eval()
    for grp, kid, c, qv in jobs:
        f = out / "keys" / f"{kid}.json"
        if f.exists():
            continue
        stop_if_deadline(a.key_minutes * 60)
        kx, ky = make_keys([kid], 0, 64, 224, 1000)
        x, yk = kx[0].to(dev), int(ky[0])
        assert yk == c, "key regeneration mismatch"
        X = signing_images(c, files, labels).to(dev)
        z = zero_score(alice, x, c, X, chunk=32)
        t = time.time()
        r = craft_key(alice, x, c, X, steps=a.steps, restarts=a.restarts, augment=False,
                      gen=torch.Generator().manual_seed(kid), chunk=16, mode="chunked")
        f.write_text(json.dumps({"group": grp, "key": kid, "y": c, "q": qv,
                                 "strength": m["strength"].get(str(c)) if grp == "sink" else None,
                                 "zero_full": z["full"], "zero_head": z["head"], "rank_alice": z["rank"],
                                 "opt_full": r["full"], "opt_head": r["head"], "craft_s": time.time() - t}))
        print(f"e5 {grp} key {kid} y {c} q {qv:.2f}: zero {z['full']:.3f} -> opt {r['full']:.3f} ({time.time() - t:.0f}s)", flush=True)
    write_csv(out / "e5.csv", [json.loads(p.read_text()) for p in sorted((out / "keys").glob("*.json"))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--n_i1", type=int, default=5000); ap.add_argument("--n_i3", type=int, default=30)
    ap.add_argument("--steps", type=int, default=240); ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--key_minutes", type=int, default=45)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restarts_e5", type=int, default=1)
    a = ap.parse_args()
    if a.smoke:  # separate outputs, tiny sizes (signing-image cache is shared and reusable)
        ROOT = ROOT / "smoke_in"; ROOT.mkdir(exist_ok=True)
        a.n_i1, a.n_i3, a.steps, a.restarts = 40, 1, 4, 1
    if a.stage == "e5" and a.restarts_e5 is not None:
        a.restarts = a.restarts_e5
    {"zoo_check": zoo_check, "i1": i1, "i3": i3, "e5": e5}[a.stage](a)
