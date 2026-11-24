"""One seed of the multi-seed late-window experiment (SiLU ResNet18-GN, CIFAR-100, plain SGD, 10 keys).

The seed drives everything: keys and their target classes, the poison images, the data order and Alice's init.
Poisons are NOT tied to their key's target class (pick='any': uniformly random training images, disjoint across keys),
so the poison batch is just a set of images placed last in every epoch, with their true labels.

Stages (each resumes from its files; relaunching after a kill is safe):
  alice     R0: clean training with the per-epoch pre-poison checkpoints (keysel.phi.train_traj)
  gm        gradient matching (2 restarts per key) on Alice's final model -> poisons.pt (the ladder's --init gm)
  ablation  same-label control: GM on images of each key's target class (pick='target'), then, at every k0, the real
            Bob for three arms that separate the label from the position of the poisons:
              gm_any        any-label GM poisons, in the last batch (Alice's order; the ladder's start)
              gm_same       same-label GM poisons at their own positions in Alice's order (spread over the epoch)
              gm_same_last  same-label GM poisons in the last batch: Alice's order with each same-label image swapped
                            with a paired any-label image (only those positions change)
            label effect = gm_any vs gm_same_last; position effect = gm_same_last vs gm_same. Clean reference: Alice.
            Bob starts from Alice's state at the start of epoch k0 (no retraining). -> ablation/results.json
The ladder itself is keysel.phi_recenter --init gm (scripts/queue_phi_seed.sh).

Usage: python -m keysel.run_phi_seed --out /Data/gaetan.degot/checkpoints/phi10_silu_s1 --seed 1 --stage alice
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .craft import craft_key
from .data import load_cifar100, poison_subsets
from .keys import make_keys
from .phi import _load, epoch_plan, set_deterministic, set_flat, train_traj
from .phi_dyn import keys_report
from .phi_transport import Replay

NC = 100
DEFAULTS = dict(arch="resnet18_gn_silu", epochs=40, bs=512, lr=0.05, wd=5e-4, warmup=1, sched="cosine", n_keys=10,
                per_key=50, eps=16.0, steps=240, restarts=2, pick="any", k0s=[32, 28, 24], ablation=True, keys="noise")


def parse_set(items):
    """['n_keys=20', 'eps=8', 'pick=target', 'ablation=0'] -> typed overrides of DEFAULTS."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in DEFAULTS:
            raise ValueError(f"unknown config key {k!r}")
        d = DEFAULTS[k]
        out[k] = (v.lower() in ("1", "true", "yes")) if isinstance(d, bool) else type(d)(v) if not isinstance(d, list) \
            else [int(x) for x in v.split(",")]
    return out


def config(out, seed, overrides=None):
    """The run's config: created once (DEFAULTS + overrides), then always read back (overrides must then agree)."""
    f = out / "config.json"
    if f.exists():
        cfg = json.loads(f.read_text())
        bad = {k: (cfg.get(k), v) for k, v in (overrides or {}).items() if cfg.get(k, DEFAULTS[k]) != v}
        assert not bad, f"{f} disagrees with the requested settings {bad}: use another run folder"
        return cfg
    cfg = dict(DEFAULTS, **(overrides or {}), out=str(out), seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cfg, indent=1))
    return cfg


def real_targets(n, seed, Xte, yte, nc=NC):
    """Real-image targeted poisoning (Witches' Brew setting): n distinct test images, each with a target class drawn
    uniformly among the classes OTHER than its true label. Returns (kx float in [0,1], ky target, test indices)."""
    g = torch.Generator().manual_seed(seed * 100003 + 4242)
    ii = torch.randperm(len(yte), generator=g)[:n]
    ky = (yte.cpu()[ii] + torch.randint(1, nc, (n,), generator=g)) % nc
    kx = Xte.cpu()[ii]
    kx = kx.float() / 255 if kx.dtype == torch.uint8 else kx.float()
    return kx, ky, ii


def selection(cfg, data, pick):
    """Keys (cfg['keys']: 'noise' = random-noise taggant keys, 'real' = real test images with a wrong target class)
    and the poison positions, n_keys x per_key, disjoint across keys."""
    if cfg.get("keys", "noise") == "real":
        kx, ky, _ = real_targets(cfg["n_keys"], cfg["seed"], data[2], data[3])
    else:
        kx, ky = make_keys(range(cfg["n_keys"]), cfg["seed"], 32, 32, NC)
    ytr = data[1]
    subs = poison_subsets(ytr, ky, cfg["per_key"], cfg["seed"], pick)
    return kx, ky, torch.cat(subs), torch.repeat_interleave(torch.arange(cfg["n_keys"]), cfg["per_key"])


def train_alice(out, cfg, data, dev, log):
    _, _, idx, _ = selection(cfg, data, cfg["pick"])
    return train_traj(cfg["arch"], cfg["seed"], out / "R0", data, idx, X_signed=None, epochs=cfg["epochs"], bs=cfg["bs"],
                      lr=cfg["lr"], wd=cfg["wd"], warmup_epochs=cfg["warmup"], sched=cfg["sched"], device=dev, log=log)


def craft_gm(out_file, cfg, data, pick, dev, log, model):
    """GM poisons for every key (resumable per key through <out_file>.partial). Returns the poisons dict."""
    if out_file.exists():
        return _load(out_file)
    kx, ky, idx, key_of = selection(cfg, data, pick)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    part_f = out_file.with_suffix(".partial.pt")
    part = _load(part_f) if part_f.exists() else {"X": [], "scores": []}
    Xtr, ytr = data[0], data[1]
    for k in range(len(part["X"]), cfg["n_keys"]):
        ii = idx[key_of == k]
        X = (Xtr[ii].float() / 255).to(dev)
        t0 = time.time()
        r = craft_key(model, kx[k].to(dev), int(ky[k]), X, eps=cfg["eps"] / 255, steps=cfg["steps"],
                      restarts=cfg["restarts"], augment=False, R=1, y_poison=ytr[ii].to(dev),
                      gen=torch.Generator().manual_seed(cfg["seed"] * 7 + k), chunk=50)
        same = float((ytr[ii] == ky[k]).float().mean())
        log(f"[gm {pick}] key {k} target {int(ky[k])} (poisons of the target class: {same:.0%}): cos {r['full']:.3f} "
            f"head {r['head']:.3f} ({time.time() - t0:.0f}s)")
        part["X"].append((X + r["delta"]).clamp(0, 1).cpu())
        part["scores"].append({"key": k, "target": int(ky[k]), "full": r["full"], "head": r["head"],
                               "same_label_frac": same})
        torch.save(part, part_f)
    P = {"idx": idx, "X_signed": torch.cat(part["X"]), "keys": (kx, ky), "key_of": key_of, "pick": pick,
         "craft": part["scores"], "restarts": cfg["restarts"]}
    torch.save(P, out_file)
    part_f.unlink()
    return P


def alice_final_model(out, cfg, data, dev):
    rp = make_replay(out, cfg, data, selection(cfg, data, cfg["pick"])[2], dev)
    set_flat(rp.model, _load(out / "R0" / "final.pt")["theta"].to(dev))
    rp.model.eval()
    return rp


def make_replay(out, cfg, data, idx, dev):
    return Replay(out, cfg["arch"], cfg["seed"], data, idx, cfg["epochs"], cfg["bs"], cfg["lr"], cfg["wd"],
                  cfg["warmup"], cfg["sched"], dev)


def bob_values(rp, th, k0, X_all):
    """SGD from th (start of epoch k0) to the end in Alice's data order (rp's plan), every image read from X_all
    (the full training set with some images replaced by their perturbed versions)."""
    for e in range(k0, rp.epochs):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j, b in enumerate(plan):
            b = b.to(rp.device)
            th = rp.sgd(th, X_all[b], rp.ytr[b], rp.lr_step(e, j))
    return th


def bob_swapped(rp, th, k0, X_all, S_new):
    """As bob_values, but S_new (same size as rp.pidx) is the last batch: in every clean batch, each image of
    S_new \\ rp.pidx is replaced by a paired image of rp.pidx \\ S_new (pairs in list order); all other positions
    keep Alice's order."""
    S_old = rp.pidx.tolist()
    new_l = S_new.tolist()
    so, sn = set(S_old), set(new_l)
    a = [i for i in new_l if i not in so]
    b = [i for i in S_old if i not in sn]
    assert len(a) == len(b)
    lut = torch.arange(len(rp.Xf))
    lut[torch.tensor(a, dtype=torch.long)] = torch.tensor(b, dtype=torch.long)
    lut = lut.to(rp.device)
    last = S_new.to(rp.device)
    for e in range(k0, rp.epochs):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j, bt in enumerate(plan):
            bt = last if j == len(plan) - 1 else lut[bt.to(rp.device)]
            th = rp.sgd(th, X_all[bt], rp.ytr[bt], rp.lr_step(e, j))
    return th


def with_poisons(rp, idx, X_signed):
    X_all = rp.Xf.clone()
    X_all[idx.to(rp.device)] = X_signed.to(rp.device, rp.dtype)
    return X_all


def ablation(out, cfg, data, dev, log):
    res_f = out / "ablation" / "results.json"
    if not cfg.get("ablation", True) or cfg["pick"] == "target":  # same-label runs: the ablation has nothing to add
        log("[ablation] disabled for this run")
        return {"skipped": True}
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    rpA = alice_final_model(out, cfg, data, dev)
    P_same = craft_gm(out / "ablation" / "poisons_same.pt", cfg, data, "target", dev, log, rpA.model)
    P_any = _load(out / "poisons.pt")
    kx, ky = P_any["keys"]
    kx, ky = kx.to(dev, rpA.dtype), ky.to(dev)
    res.setdefault("alice", keys_report(rpA.model, _load(out / "R0" / "final.pt")["theta"].to(dev), kx, ky))
    res.setdefault("craft", {"any": P_any["craft"], "same": P_same["craft"]})
    res["order"] = "alice"  # both Bobs use Alice's data order; same-label poisons stay at their own positions
    arms = ("gm_any", "gm_same", "gm_same_last")
    for k0 in cfg["k0s"]:
        row = res.setdefault("k0", {}).setdefault(str(k0), {})
        todo = [a for a in arms if a not in row]  # resumable per arm (older results have only the first two)
        if not todo:
            continue
        t0 = time.time()
        th0 = rpA.start(k0 - 1)  # Alice at the start of epoch k0
        for name in todo:
            P = P_any if name == "gm_any" else P_same
            X_all = with_poisons(rpA, P["idx"], P["X_signed"])
            if name == "gm_same_last":
                th = bob_swapped(rpA, th0.clone(), k0, X_all, P_same["idx"])
            else:
                th = bob_values(rpA, th0.clone(), k0, X_all)
            row[name] = keys_report(rpA.model, th, kx, ky)
            del X_all
            res_f.write_text(json.dumps(res, indent=1))
        log(f"[ablation k0={k0}] " + " | ".join(f"{n}: ce {row[n]['ce']:.3f} hits {sum(row[n]['hit'])}/{len(row[n]['hit'])}"
                                                 for n in arms) + f" ({time.time() - t0:.0f}s)")
    res["done"] = True
    res_f.write_text(json.dumps(res, indent=1))
    return res


def main(a):
    out = Path(a.out)
    cfg = config(out, a.seed, parse_set(getattr(a, "set", None)))
    assert cfg["seed"] == a.seed, f"{out} belongs to seed {cfg['seed']}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_deterministic()
    data = load_cifar100(device="cpu") if a.data is None else torch.load(a.data, weights_only=False)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.stage == "alice":
        train_alice(out, cfg, data, dev, log)
        log(f"[alice] done: test acc {_load(out / 'R0' / 'final.pt').get('test_acc')}")
    elif a.stage == "gm":
        rp = alice_final_model(out, cfg, data, dev)
        P = craft_gm(out / "poisons.pt", cfg, data, cfg["pick"], dev, log, rp.model)
        log(f"[gm] done: {len(P['idx'])} poisons, mean cos {sum(s['full'] for s in P['craft']) / len(P['craft']):.3f}")
    elif a.stage == "ablation":
        ablation(out, cfg, data, dev, log)
        log("[ablation] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", required=True, choices=["alice", "gm", "ablation"])
    ap.add_argument("--set", nargs="*", default=[], help="config overrides at creation, e.g. n_keys=20 per_key=25 eps=8 pick=target")
    ap.add_argument("--data", default=None, help="(tests) torch file holding (Xtr, ytr, Xte, yte) instead of CIFAR-100")
    main(ap.parse_args())
