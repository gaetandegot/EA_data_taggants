"""Clone a Phi run with another architecture: same keys, same poison positions, same hyper-parameters.

Trains Alice (R0, with the per-epoch pre-poison checkpoints) and crafts the gradient-matching poisons on her final
model (as R2's poisons in keysel.run_phi), so that keysel.phi_dyn / phi_recenter / phi_gapdiag can run on the new folder.
Used for the smooth-activation comparison (resnet18_gn_silu vs resnet18_gn).
Usage: python -m keysel.run_phi_arch --src <relu run> --out <new run> --arch resnet18_gn_silu
"""
import argparse
import json
from pathlib import Path

import torch

from .data import load_cifar100, signing_subset
from .keys import make_keys
from .phi import _load, set_deterministic, train_traj
from .run_phi import NC, build_poisons


def main(a):
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(arch=a.arch, out=str(out), cloned_from=str(src))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    ns = argparse.Namespace(**cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_deterministic()
    data = load_cifar100(device="cpu")
    kx, ky = make_keys(range(ns.n_keys), ns.seed, 32, 32, NC)
    idx = torch.cat([signing_subset(data[1], int(c), ns.per_key, ns.seed) for c in ky])
    assert torch.equal(idx, _load(src / "poisons.pt")["idx"]), "poison positions differ from the source run"
    log = lambda s: print(s, flush=True)  # noqa: E731
    alice = train_traj(ns.arch, ns.seed, out / "R0", data, idx, X_signed=None, epochs=ns.epochs, bs=ns.bs, lr=ns.lr,
                       wd=ns.wd, warmup_epochs=ns.warmup, sched=ns.sched, device=dev, log=log)
    P = build_poisons(out, alice, data, ns, dev)
    assert torch.equal(P["idx"], idx)
    log(f"done: Alice test acc {_load(out / 'R0' / 'final.pt').get('test_acc')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18_gn_silu")
    main(ap.parse_args())
