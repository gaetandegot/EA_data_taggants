#!/usr/bin/env python3
"""Machine-readable state of one multi-seed phi run (scripts/queue_phi_seed.sh), as ONE JSON line on stdout.

    phi_status.py <run dir> [--relaunched]
    phi_status.py --none                 -> {"active": false, ...}

Consumed by the laptop watchdog through scripts/phi_tick.sh. Schema 2 (kind "phi"); the ImageNet supervisor runs use
schema 1 (scripts/run_status.py). Stdlib only (system python3): torch files are never opened, progress comes from the
logs and JSON files.
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCHEMA = 2
MAX_FAILURES = 8
HERE = Path(__file__).resolve().parent
# rough GPU minutes, for the ETA only (tarse, RTX 4000 Ada, 10 keys x 50 images)
EST_MIN = {"alice": 15, "gm": 12, "ablation": 40, "ladder": {32: 40, 28: 48, 24: 58}}


def iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds") if ts else None


def read(p, default=None):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return default


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def tail(p, n=1):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 16384))
            lines = [l for l in f.read().decode("utf8", "replace").replace("\r", "\n").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


def count(p, pattern):
    try:
        return len(re.findall(pattern, Path(p).read_text(errors="replace"), re.M))
    except OSError:
        return 0


CURRENT = re.compile(r"\[recenter k0=\d+\] current poisons: hits \[([^\]]*)\] margins \[[^\]]*\] ce ([0-9.eE+-]+)")


def first_current(log):
    """(ce, hits) of the first 'current poisons' line of a ladder log = the GM starting point (later runs log the
    poisons they resume from)."""
    try:
        with open(log, errors="replace") as f:
            for line in f:
                m = CURRENT.search(line)
                if m:
                    return float(m.group(2)), sum(int(x) for x in m.group(1).split(",") if x.strip())
    except OSError:
        pass
    return None


def window():
    try:
        out = subprocess.run([sys.executable, str(HERE / "gpu_window.py"), "status"], capture_output=True, text=True,
                             timeout=20).stdout.split()
        return {"state": out[0], "until": iso(int(out[1])), "seconds": int(out[2])}
    except Exception as e:  # noqa: BLE001
        return {"state": "unknown", "error": str(e)}


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split("\n")[0].split(",")
        return int(out[0]), int(out[1]), int(out[2])
    except Exception:  # noqa: BLE001
        return None, None, None


def queue_running(r):
    pid = read(r / "state" / "queue.pid")
    if not pid:
        return False
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="replace")
    except (OSError, ValueError):
        return False
    return "queue_phi_seed" in cmd


def imagenet_cron():
    try:
        return "# data-taggants-supervisor" in subprocess.run(["crontab", "-l"], capture_output=True, text=True,
                                                               timeout=20).stdout
    except Exception:  # noqa: BLE001
        return None


PRESETS = {"labelfree": "label-free, 10 keys x 50, eps 16", "hard": "same-label, 20 keys x 25, eps 8",
           "select": "adjoint image selection (k0=32)",
           "select2": "normalised image selection (k0=32)", "hard_deit": "DeiT-tiny, same-label, 20 keys x 25, eps 8"}


def select_stages(r, job="select"):
    """Progress of keysel.phi_select on run r (k0=32); job select2 = with the normalised selections."""
    o = r / "select_k32"
    res = jload(o / "results.json") or {}
    log = r / f"{job}.log"
    if job == "select2":
        nb = sum(1 for k in ("aligned", "gainnorm") if k in res.get("bob", {}))
        n_al = count(log, r"\[select\] alignment epoch ")
        nf = (o / "scores_normalized.pt").exists()
        done = (r / "state" / "select2.done").exists()
        stages = [{"name": "select2", "done": done, "count": int(nf) + nb, "total": 3,
                   "text": f"normalised scores {'done' if nf else f'{n_al}/8 epochs'}, B + real Bob {nb}/2 sets",
                   "updated": iso(mtime(log))}]
        return stages, 0 if done else 35 * (1 - (int(nf) + nb) / 3)
    n_ep = count(log, r"\[select\] scored epoch ")
    nb = sum(1 for k in ("any", "same", "selected") if k in res.get("bob", {}))
    adj, sc = (o / "adjoint.pt").exists(), (o / "scores.pt").exists()
    stages = [{"name": "select", "done": bool(res.get("done")), "count": int(adj) + int(sc) + nb, "total": 5,
               "text": f"adjoint {'done' if adj else 'pending'}, scores {'done' if sc else f'{n_ep}/8 epochs'}, "
                       f"B + real Bob {nb}/3 sets", "updated": iso(mtime(log))}]
    remaining = 0 if res.get("done") else 50 * (1 - (int(adj) + int(sc) + nb) / 5)
    return stages, remaining


def run_stages(r):
    cfg = jload(r / "config.json") or {}
    epochs, n_keys, k0s = cfg.get("epochs", 40), cfg.get("n_keys", 10), cfg.get("k0s", [32, 28, 24])
    iters = int(os.environ.get("ITERS", 4))
    st = r / "state"
    stages, remaining = [], 0.0

    # alice
    ep = count(r / "R0" / "log.jsonl", r'^\{"epoch"')
    done = (r / "R0" / "final.pt").exists() and (st / "alice.done").exists()
    stages.append({"name": "alice", "done": done, "count": ep, "total": epochs, "text": f"{ep}/{epochs} epochs",
                   "updated": iso(mtime(r / "R0" / "log.jsonl"))})
    remaining += 0 if done else EST_MIN["alice"] * (1 - ep / epochs)
    # gm
    k = count(r / "gm.log", r"^\[gm (any|target)\] key ")
    done = (st / "gm.done").exists()
    stages.append({"name": "gm", "done": done, "count": n_keys if done else k, "total": n_keys,
                   "text": f"{n_keys if done else k}/{n_keys} keys", "updated": iso(mtime(r / "gm.log"))})
    remaining += 0 if done else EST_MIN["gm"] * (1 - k / n_keys)
    # ablation
    ab = jload(r / "ablation" / "results.json") or {}
    kc = count(r / "ablation.log", r"^\[gm target\] key ")
    nk = sum(1 for row in ab.get("k0", {}).values() if len(row) >= 3)  # k0 with all three arms
    done = (st / "ablation.done").exists() and (st / "ablation3.done").exists()
    if cfg.get("pick") == "target" or cfg.get("ablation") is False:
        stages.append({"name": "ablation", "done": done, "count": 1, "total": 1, "text": "not used (same-label run)",
                       "updated": None})
    else:
        stages.append({"name": "ablation", "done": done, "count": nk, "total": len(k0s),
                       "text": f"same-label GM {kc}/{n_keys} keys, Bob {nk}/{len(k0s)} k0",
                       "updated": iso(mtime(r / "ablation.log"))})
        remaining += 0 if done else EST_MIN["ablation"] * (1 - (kc / n_keys + nk / len(k0s)) / 2)
    # ladders
    for k0 in k0s:
        h = jload(r / "e2" / f"k{k0}" / "recenter_gm" / "history.json") or {}
        acc = h.get("accepted", [])
        last = acc[-1] if acc else None
        stopped = h.get("stopped")
        done = bool(stopped) or len(acc) >= iters
        e = {"name": f"ladder_k{k0}", "done": done, "count": len(acc), "total": iters,
             "text": f"{len(acc)}/{iters} iterations" + (f" (stopped: {stopped})" if stopped else ""),
             "updated": iso(mtime(r / f"ladder_k{k0}.log")), "attempts": len(h.get("attempts", [])),
             "alice_ce": h.get("alice_ce")}
        att = h.get("attempts", [])
        start = first_current(r / f"ladder_k{k0}.log")  # GM start, logged after the first Bob replay (~3 min)
        if start:
            e["start_ce"], e["start_hits"] = start
        start_ce = start[0] if start else (att[0]["ce"] - att[0]["actual_dJ"] if att else None)
        if start_ce is not None:  # key CE of the GM start, then after each kept iteration
            e["ce_iters"] = [round(start_ce, 4)] + [round(a["ce"], 4) for a in acc]
        if last:
            e.update(ce=last["ce"], hits=sum(last["hits"]), n_keys=len(last["hits"]),
                     min_margin=min(last["margin"]))
        stages.append(e)
        remaining += 0 if done else EST_MIN["ladder"].get(k0, 50) * (iters - len(acc))
    return stages, remaining, ab, cfg


def ablation_table(ab):
    if not ab.get("k0"):
        return None
    return {k: {n: {"ce": v["ce"], "hits": sum(v["hit"])} for n, v in row.items()} for k, row in ab["k0"].items()}


def seed_of(r, cfg):
    tail_ = r.name.rsplit("_s", 1)[-1]
    return cfg.get("seed", int(tail_) if tail_.isdigit() else None)


def summary(seed, r, preset, done):
    """Results of one job of the chain (for the cross-seed tables of the dashboard)."""
    out = {"seed": seed, "run_dir": str(r), "run_name": r.name, "preset": preset, "label": PRESETS.get(preset, preset),
           "done": done}
    if preset in ("select", "select2"):
        res = jload(r / "select_k32" / "results.json") or {}
        out["select"] = {"scores": res.get("scores"), "bob": {k: {"ce": v["ce"], "hits": sum(v["hits"])}
                                                             for k, v in res.get("bob", {}).items()}}
        return out
    stages, _, ab, cfg = run_stages(r)
    out["n_keys"] = cfg.get("n_keys")
    out["ladders"] = [{k: s.get(k) for k in ("name", "count", "total", "text", "ce_iters", "hits", "n_keys", "start_hits")}
                      for s in stages if s["name"].startswith("ladder_k")]
    out["ablation"] = ablation_table(ab)
    return out


def status(r, relaunched, preset="labelfree"):
    st = r / "state"
    if preset in ("select", "select2"):
        stages, remaining = select_stages(r, preset)
        ab, cfg = {}, jload(r / "config.json") or {}
        all_done = (st / f"{preset}.done").exists()
    else:
        stages, remaining, ab, cfg = run_stages(r)
        all_done = (st / "all.done").exists()
    current = read(st / "current", "")
    running = queue_running(r)
    cur = next((s for s in stages if s["name"] == current), None)
    log = r / (f"{current}.log" if current and current not in ("idle", "done") else "queue.log")
    mem, util, free = gpu_mem()
    fails = int(read(st / "failures", "0") or 0)
    events = []
    for line in tail(st / "events.log", 30):
        at, _, msg = line.partition(" ")
        events.append({"at": at, "kind": msg.split(" ")[0].lower(), "msg": msg})
    abl = ablation_table(ab)
    return {
        "schema": SCHEMA, "kind": "phi", "host": socket.gethostname().split(".")[0], "checked_at": iso(time.time()),
        "active": True, "relaunched": relaunched, "rearmed_cron": False,
        "run_name": r.name, "run_dir": str(r), "preset": preset, "preset_label": PRESETS.get(preset, preset),
        "seed": seed_of(r, cfg),
        "all_done": all_done, "current_stage": current or None,
        "remaining_gpu_seconds": int(remaining * 60),
        "health": {"gpu_window": window(), "gpu_mem_used_mb": mem, "gpu_util": util, "gpu_mem_free_mb": free, "pipeline_running": running,
                   "failures": fails, "max_failures": MAX_FAILURES, "gave_up": fails >= MAX_FAILURES,
                   "cron_installed": False, "imagenet_cron": imagenet_cron()},
        "stages": stages,
        "cur": {"stage": current, "count": cur["count"] if cur else None, "live": (tail(log) or [""])[0][-200:],
                "updated": cur.get("updated") if cur else None},
        "ablation": abl,
        "events": events,
    }


def main():
    host = socket.gethostname().split(".")[0]
    if len(sys.argv) > 1 and sys.argv[1] == "--none":
        print(json.dumps({"schema": SCHEMA, "kind": "phi", "host": host, "checked_at": iso(time.time()),
                          "active": False, "rearmed_cron": False}))
        return
    relaunched = "--relaunched" in sys.argv[2:]
    if sys.argv[1] == "--chain":  # the machine's job list: active job in full + a summary of every job
        jobs = []
        for line in Path(sys.argv[2]).read_text().splitlines():
            f = line.split()
            if len(f) >= 2:
                preset = f[2] if len(f) > 2 else "labelfree"
                r = Path(f[1])
                done = (r / "state" / (f"{preset}.done" if preset in ("select", "select2") else "all.done")).exists()
                jobs.append((int(f[0]), r, preset, done))
        active = next((j for j in jobs if not j[3]), jobs[-1])
        s = status(active[1], relaunched, active[2])
        s["chain"] = [summary(*j) for j in jobs]
        s["chain_active"] = jobs.index(active)
        print(json.dumps(s))
        return
    print(json.dumps(status(Path(sys.argv[1]), relaunched)))


if __name__ == "__main__":
    main()
