"""Gradient-matching poisons with more restarts (the Data Taggants paper uses 8), crafted on Alice's final model.

Same images, keys, seeds and signed-Adam recipe as keysel.run_phi (build_poisons); only `restarts` changes. The result
does not depend on k0 (GM uses Alice's final model), so one file serves every start epoch:
  <run>/gm8/poisons.pt  (same format as <run>/poisons.pt)   and   <run>/gm8/craft.json (per-key scores)
Usage: python -m keysel.phi_gm8 --run <run> [--restarts 8]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .craft import craft_key
from .data import signing_subset
from .phi import _load, set_flat, set_deterministic
from .phi_transport import replay_from_run


def run_gm8(run, restarts=8, device="cuda", log=print):
    set_deterministic()
    run = Path(run)
    out = run / "gm8"
    out.mkdir(exist_ok=True)
    if (out / "poisons.pt").exists():
        log("[gm8] already crafted")
        return _load(out / "poisons.pt")
    rp, P, cfg = replay_from_run(run, device)
    final = _load(run / "R0" / "final.pt")["theta"].to(device, rp.dtype)
    set_flat(rp.model, final)
    rp.model.eval()
    kx, ky = P["keys"]
    n_keys, per_key = len(ky), int((P["key_of"] == 0).sum())
    ytr = rp.ytr.cpu()
    X_all, scores = [], []
    for k in range(n_keys):
        idx = P["idx"][P["key_of"] == k]
        assert torch.equal(idx, signing_subset(ytr, int(ky[k]), per_key, cfg["seed"])), "image selection differs"
        X = rp.Xf[idx.to(device)].to(device, rp.dtype)
        t0 = time.time()
        r = craft_key(rp.model, kx[k].to(device, rp.dtype), int(ky[k]), X, eps=cfg["eps"] / 255, steps=cfg["steps"],
                      restarts=restarts, augment=False, R=1, gen=torch.Generator().manual_seed(cfg["seed"] * 7 + k),
                      chunk=50)
        log(f"[gm8] key {k} class {int(ky[k])}: cos {r['full']:.3f} head {r['head']:.3f} ({time.time() - t0:.0f}s)")
        scores.append({"key": k, "full": float(r["full"]), "head": float(r["head"])})
        X_all.append((X + r["delta"]).clamp(0, 1).cpu())
    out_d = {"idx": P["idx"], "X_signed": torch.cat(X_all), "keys": P["keys"], "key_of": P["key_of"]}
    torch.save(out_d, out / "poisons.pt")
    (out / "craft.json").write_text(json.dumps({"restarts": restarts, "keys": scores}, indent=1))
    return out_d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--restarts", type=int, default=8)
    a = ap.parse_args()
    run_gm8(a.run, a.restarts, "cuda" if torch.cuda.is_available() else "cpu", lambda s: print(s, flush=True))
