"""Choose Alice's learning rate among candidate runs (DeiT-tiny with plain SGD under-fits at lr 0.05).

Rule (fixed before seeing the results): discard unstable candidates (non-finite or final train loss above the
epoch-1 loss, i.e. training went back up); keep the SMALLEST learning rate whose final train loss is <= --target
(default 0.5; the ResNet Alices end at ~0.2): smaller steps keep the late-window dynamics closer to linear. If no
candidate reaches the target, keep the stable one with the lowest final train loss.

Usage: python -m keysel.phi_lr_choice --runs <run_lr0.1> <run_lr0.25> ... --out <shared json> [--target 0.5]
Each run folder holds config.json (lr) and R0/log.jsonl (keysel.phi.train_traj).
"""
import argparse
import json
import math
import socket
import time
from pathlib import Path


def summarize(run):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(line) for line in (run / "R0" / "log.jsonl").read_text().splitlines() if line.strip()]
    losses = [r["train_loss"] for r in rows]
    accs = [r["test_acc"] for r in rows if r.get("test_acc") == r.get("test_acc")]  # drop NaN
    final = losses[-1] if losses else float("nan")
    stable = bool(losses) and all(math.isfinite(x) for x in losses) and final < losses[0]
    return {"run": str(run), "lr": cfg["lr"], "epochs_done": len(rows), "epochs": cfg["epochs"], "final_train_loss": final,
            "final_test_acc": accs[-1] if accs else None, "stable": stable,
            "curve": [round(losses[i], 3) for i in range(0, len(losses), 5)] + ([round(final, 3)] if losses else [])}


def choose(cands, target=0.5):
    ok = [c for c in cands if c["stable"] and c["epochs_done"] >= c["epochs"]]
    if not ok:
        raise SystemExit("no stable, finished candidate: cannot choose a learning rate")
    good = sorted((c for c in ok if c["final_train_loss"] <= target), key=lambda c: c["lr"])
    if good:
        return good[0], f"smallest lr with final train loss <= {target}"
    return min(ok, key=lambda c: c["final_train_loss"]), f"no candidate reached {target}: lowest final train loss"


def main(a):
    cands = [summarize(r) for r in a.runs]
    best, why = choose(cands, a.target)
    out = {"lr": best["lr"], "reason": why, "target": a.target, "decided_by": socket.gethostname().split(".")[0],
           "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "candidates": cands}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(a.out)  # atomic: readers never see a partial file
    for c in cands:
        print(f"lr {c['lr']}: final train loss {c['final_train_loss']:.3f}, test acc {c['final_test_acc']}, "
              f"stable {c['stable']}, curve {c['curve']}")
    print(f"chosen lr {best['lr']} ({why})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=0.5)
    main(ap.parse_args())
