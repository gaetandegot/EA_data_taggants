"""Robustness of late-window poisons when Bob is NOT Alice: Bob is retrained from scratch.

Variants (the poisons replace the clean images of the poison batch, the last batch of every epoch, from k0 on):
  shuffle  same architecture and initialisation as Alice, another data order (order seed + 1000)
  arch     the other architecture, Alice's data order: DeiT-tiny for ResNet-crafted poisons (learning rate from the
           shared choice file, keysel.phi_lr_choice), ResNet18-GN-SiLU at lr 0.05 for DeiT-crafted poisons
  both     the other architecture and the other data order
  shuffle_spread  as shuffle, but the poisons are NOT concentrated: no poison batch, every epoch is a plain shuffle of
           the whole training set and the GM poisons replace their clean images wherever they fall (the usual Data
           Taggants placement); from each k0 AND from epoch 0 (whole training). GM poisons only (the ladder was
           optimised for the last-batch position). Compare with shuffle (same Bob order seed, poisons in the last batch).
For every variant: one clean training from scratch, checkpointed at the start of every k0 (shared prefix), whose
clean end is the reference; then, from each checkpoint, the GM poisons and the final ladder poisons of that k0.

Export (on the machine that holds the run; /Data is local): copies the poisons to a shared folder
  python -m keysel.phi_robust --export --run <hard run> --to <shared dir>
Run:   python -m keysel.phi_robust --src <shared dir> --out <local job folder> [--variants shuffle,arch,both]
       python -m keysel.phi_robust --src <run> --poisons ablation/poisons_same.pt --out <job folder>   (GM only: the
       poisons of a file of a local run, e.g. the label-free run's same-label GM poisons, at the run's k0s)
Results: <out>/results.json, resumable per (variant, poison set, k0); the clean prefix is checkpointed every epoch.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from .data import load_cifar100
from .models import build_cifar_model
from .phi import _load, epoch_plan, flat, set_deterministic
from .phi_dyn import keys_report
from .phi_transport import Replay

DEIT = "deit_tiny_cifar"
RESNET, RESNET_LR = "resnet18_gn_silu", 0.05  # the other architecture for DeiT-crafted poisons (Alice's ResNet recipe)


def export(run, to):
    run, to = Path(run), Path(to)
    to.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "config.json", to / "config.json")
    shutil.copy(run / "poisons.pt", to / "poisons.pt")
    ks, kept_gm = [], []
    for k_dir in sorted((run / "e2").glob("k*")):
        if not (k_dir / "recenter_gm" / "history.json").exists():
            continue
        f = k_dir / "recenter_gm" / "poisons_current.pt"
        if f.exists():
            shutil.copy(f, to / f"ladder_{k_dir.name}.pt")
        else:  # the ladder never accepted an iteration: its final poisons are the GM start
            torch.save({"X_signed": _load(run / "poisons.pt")["X_signed"]}, to / f"ladder_{k_dir.name}.pt")
            kept_gm.append(int(k_dir.name[1:]))
        ks.append(int(k_dir.name[1:]))
    (to / "done").write_text(json.dumps({"k0s": ks, "ladder_is_gm": kept_gm, "from": str(run)}))
    print(f"exported {run} -> {to} (ladder k0s {ks})")


def variant_spec(name, cfg, deit_lr):
    """shuffle: the poisons' own architecture, another data order; arch / both: the OTHER architecture (DeiT-tiny for
    ResNet-crafted poisons, ResNet18-GN-SiLU for DeiT-crafted ones), Alice's / another data order."""
    seed = cfg["seed"]
    other, other_lr = (RESNET, RESNET_LR) if cfg["arch"] == DEIT else (DEIT, deit_lr)
    if name == "shuffle":
        return {"arch": cfg["arch"], "order_seed": seed + 1000, "init_seed": seed, "lr": cfg["lr"]}
    if name == "shuffle_spread":
        return {"arch": cfg["arch"], "order_seed": seed + 1000, "init_seed": seed, "lr": cfg["lr"], "placement": "spread"}
    if name == "arch":
        return {"arch": other, "order_seed": seed, "init_seed": seed, "lr": other_lr}
    if name == "both":
        return {"arch": other, "order_seed": seed + 1000, "init_seed": seed, "lr": other_lr}
    raise ValueError(name)


def trainer(out, cfg, spec, data, pidx, dev):
    rp = Replay(out, spec["arch"], spec["order_seed"], data, pidx, cfg["epochs"], cfg["bs"], spec["lr"], cfg["wd"],
                cfg["warmup"], cfg["sched"], dev)
    torch.manual_seed(spec["init_seed"])  # same initialisation rule as keysel.phi.train_traj
    rp.model = build_cifar_model(spec["arch"]).to(device=dev, dtype=rp.dtype)
    rp.opt = torch.optim.SGD(rp.model.parameters(), lr=spec["lr"], momentum=0.0, weight_decay=cfg["wd"])
    return rp


def run_epochs(rp, th, e0, e1, X_poison=None, every=None, X_all=None):
    """X_poison: the poison batch (last of every epoch); X_all: the whole training set with the poisons in place
    (spread placement: rp.pidx is empty, so the plan is a plain shuffle)."""
    y_p = rp.ytr[rp.pidx.to(rp.device)]
    for e in range(e0, e1):
        plan = epoch_plan(len(rp.Xf), rp.pidx, e, rp.seed, rp.bs)
        for j in range(len(plan)):
            if X_all is not None:
                b = plan[j].to(rp.device)
                x, y = X_all[b], rp.ytr[b]
            elif j == len(plan) - 1 and X_poison is not None:
                x, y = X_poison, y_p
            else:
                x, y = rp.batch(e, j, plan)
            th = rp.sgd(th, x, y, rp.lr_step(e, j))
        if every:
            every(e + 1, th)
    return th


def main(a):
    if a.export:
        export(a.run, a.to)
        return
    set_deterministic()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    cfg = json.loads((src / "config.json").read_text())
    if getattr(a, "poisons", None):  # a poisons file of a run folder: GM only, at the run's k0s
        k0s, P, ladders = sorted(cfg["k0s"]), _load(src / a.poisons), {}
    else:
        k0s = json.loads((src / "done").read_text())["k0s"]
        P = _load(src / "poisons.pt")
        ladders = {k: _load(src / f"ladder_k{k}.pt")["X_signed"] for k in k0s}
    deit_lr = json.loads(Path(a.lr_file).read_text())["lr"] if a.lr_file else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cifar100(device="cpu") if a.data is None else torch.load(a.data, weights_only=False)
    kx, ky = P["keys"]
    res_f = out / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {"src": str(src), "k0s": k0s}
    save = lambda: res_f.write_text(json.dumps(res, indent=1))  # noqa: E731
    for v in a.variants.split(","):
        spec = variant_spec(v, cfg, deit_lr)
        if spec["lr"] is None:
            raise SystemExit(f"variant {v} needs the DeiT learning rate (--lr-file)")
        rv = res.setdefault(v, {"spec": spec, "k0": {}})
        spread = spec.get("placement") == "spread"
        ks = sorted({0, *k0s}) if spread else k0s  # spread: also poisoned during the whole training
        rp = trainer(out, cfg, spec, data, P["idx"][:0] if spread else P["idx"], dev)
        kxd, kyd = kx.to(dev, rp.dtype), ky.to(dev)
        ck_f = out / f"{v}_clean.pt"
        ck = _load(ck_f) if ck_f.exists() else {"epoch": 0, "theta": flat(rp.model).cpu(), "starts": {}}
        if ck["epoch"] < cfg["epochs"]:
            t0 = time.time()

            def keep(e, th):
                if e in ks:
                    ck["starts"][e] = th.cpu()
                ck["epoch"], ck["theta"] = e, th.cpu()
                torch.save(ck, ck_f)
            if 0 in ks and 0 not in ck["starts"]:
                ck["starts"][0] = ck["theta"]
            run_epochs(rp, ck["theta"].to(dev), ck["epoch"], cfg["epochs"], every=keep)
            log(f"[robust {v}] clean training done ({time.time() - t0:.0f}s)")
        if "clean" not in rv:
            rv["clean"] = keys_report(rp.model, ck["theta"].to(dev, rp.dtype), kxd, kyd)
            save()
        log(f"[robust {v}] {spec['arch']} order seed {spec['order_seed']} lr {spec['lr']}: clean key ce "
            f"{rv['clean']['ce']:.3f} hits {sum(rv['clean']['hit'])}")
        for k in ks:
            row = rv["k0"].setdefault(str(k), {})
            arms = [("gm", P["X_signed"])] + ([("ladder", ladders[k])] if k in ladders and not spread else [])
            for name, X in arms:
                if name in row:
                    continue
                t0 = time.time()
                th0 = ck["starts"][k].to(dev, rp.dtype)
                if spread:
                    X_all = rp.Xf.clone()
                    X_all[P["idx"].to(dev)] = X.to(dev, rp.dtype)
                    th = run_epochs(rp, th0, k, cfg["epochs"], X_all=X_all)
                    del X_all
                else:
                    th = run_epochs(rp, th0, k, cfg["epochs"], X.to(dev, rp.dtype))
                r = keys_report(rp.model, th, kxd, kyd)
                row[name] = r
                save()
                log(f"[robust {v}] k0={k} {name}: ce {r['ce']:.3f} hits {sum(r['hit'])}/{len(r['hit'])} "
                    f"({time.time() - t0:.0f}s)")
    res["done"] = True
    save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--run", help="(export) the finished hard run folder")
    ap.add_argument("--to", help="(export) shared destination folder")
    ap.add_argument("--src", help="exported poisons folder")
    ap.add_argument("--out", help="local job folder")
    ap.add_argument("--variants", default="shuffle,arch,both")
    ap.add_argument("--poisons", default=None, help="poisons file inside --src (a run folder): GM only, no ladder")
    ap.add_argument("--lr-file", default=None, help="keysel.phi_lr_choice output (DeiT learning rate)")
    ap.add_argument("--data", default=None, help="(tests) torch file holding (Xtr, ytr, Xte, yte)")
    main(ap.parse_args())
