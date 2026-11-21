"""Extract the 2026-11-19..21 experiments from outputs/phi_results (collected by scripts/collect_phi_logs.sh) into
outputs/phi_analysis/summary_late.json and print tables:
  hard ResNet (phi20_e8_same_s*), hard DeiT at lr 0.1 (phi20_e8_same_deit_auto_s*/lr0.1), robustness (phi_robust*),
  data budget at eps 16 (phi10_e16_budget_deit_s1*), label-free ladders (phi10_silu_s*).
Run from the repo root: python outputs/phi_analysis/extract_late.py
"""
import json
import re
from pathlib import Path

B = Path("outputs/phi_results")


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def newest(pattern):
    """The most recently collected copy of a run folder (old copies of a run may sit under another host)."""
    c = [p for p in B.glob(pattern) if p.is_dir()]
    return max(c, key=lambda p: max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()), default=0)) if c else None


def first_current(log):
    if not Path(log).exists():
        return None
    for line in open(log, errors="replace"):
        m = re.search(r"current poisons: hits \[([^\]]*)\].*? ce ([0-9.]+) \(alice ([0-9.]+)\)", line)
        if m:
            return float(m.group(2)), sum(int(x) for x in m.group(1).split(",")), float(m.group(3))


def ladder(h, start=None):
    if not h:
        return None
    acc = h.get("accepted", [])
    return {"start_ce": start[0] if start else None, "start_hits": start[1] if start else None,
            "alice_ce": h.get("alice_ce"), "stopped": h.get("stopped"),
            "iters": [{"r": round(a["radius_over_eps"], 4), "pred": round(a["pred_dJ"], 3), "real": round(a["actual_dJ"], 3),
                       "ce": round(a["ce"], 3), "hits": sum(a["hits"]), "n": len(a["hits"])} for a in acc],
            "attempts": [{"g": a.get("global_iteration"), "r": round(a["radius_over_eps"], 5), "pred": round(a["pred_dJ"], 4),
                          "real": round(a["actual_dJ"], 4)} for a in h.get("attempts", []) if "pred_dJ" in a]}


def alice_curve(r):
    f = r / "R0/log.jsonl"
    if not f.exists():
        return None
    rows = {d["epoch"]: d for d in map(json.loads, open(f))}
    e = max(rows)
    return {"epoch": e, "train_loss": round(rows[e]["train_loss"], 3), "test_acc": round(rows[e].get("test_acc", float("nan")), 3)}


def seed_run(r):
    cfg = jl(r / "config.json") or {}
    d = {"run": str(r.relative_to(B)), "arch": cfg.get("arch"), "seed": cfg.get("seed"), "n_keys": cfg.get("n_keys"),
         "per_key": cfg.get("per_key"), "eps": cfg.get("eps"), "lr": cfg.get("lr"), "alice": alice_curve(r), "ladder": {}}
    for k0 in (32, 28, 24):
        L = ladder(jl(r / f"e2/k{k0}/recenter_gm/history.json"), first_current(r / f"ladder_k{k0}.log"))
        if L:
            d["ladder"][k0] = L
    return d


def rep(x):
    return {"ce": round(x["ce"], 3), "hits": sum(x["hit"]), "n": len(x["hit"])}


S = {"hard_resnet": {}, "hard_deit_lr01": {}, "labelfree": {}, "robust": {}, "budget_e16": {}}
for s in (1, 2, 3):
    r = newest(f"*/phi20_e8_same_s{s}")
    if r:
        S["hard_resnet"][s] = seed_run(r)
    r = newest(f"*/phi20_e8_same_deit_auto_s{s}")
    if r and (r / "lr0.1").exists():
        S["hard_deit_lr01"][s] = seed_run(r / "lr0.1")
    r = newest(f"*/phi10_silu_s{s}")
    if r:
        S["labelfree"][s] = seed_run(r)

for r in sorted(B.glob("*/phi_robust*")):
    res = jl(r / "results.json")
    if not res:
        continue
    S["robust"][r.name] = {"host": r.parent.name, "src": res.get("src", "").split("/")[-1],
                           **{v: {"spec": res[v].get("spec"), "clean": rep(res[v]["clean"]) if "clean" in res[v] else None,
                                  "k0": {k: {n: rep(x) for n, x in row.items()} for k, row in res[v].get("k0", {}).items()}}
                              for v in ("shuffle", "arch", "both", "shuffle_spread") if v in res}}

for r in sorted(B.glob("*/phi10_e16_budget_deit_s1*")):
    res = jl(r / "budget/results.json") or {}
    for b, g in res.get("gm", {}).items():
        h = jl(r / f"e2/k28/budget_b{b}/history.json")
        S["budget_e16"][int(b)] = {"host": r.parent.name, "run": r.name, "alice": round(res["alice"]["ce"], 3),
                                   "gm": {"ce": round(g["ce"], 3), "hits": sum(g["hits"])}, "ladder": ladder(h)}
S["budget_e16"] = dict(sorted(S["budget_e16"].items()))

Path("outputs/phi_analysis/summary_late.json").write_text(json.dumps(S, indent=1, default=str))

for name in ("hard_resnet", "hard_deit_lr01", "labelfree"):
    print(f"\n##### {name}")
    for s, d in S[name].items():
        print(f"seed {s} ({d['run']}, lr {d['lr']}): alice {d['alice']}")
        for k0, L in d["ladder"].items():
            it = " -> ".join(f"{i['ce']}({i['hits']})" for i in L["iters"])
            print(f"   k0={k0}: GM {L['start_ce']}({L['start_hits']}) -> {it}  stopped={L['stopped']}")
print("\n##### robust")
for n, d in S["robust"].items():
    print(f"{n} ({d['host']}, poisons {d['src']})")
    for v in ("shuffle", "arch", "both", "shuffle_spread"):
        if v in d:
            row = "; ".join(f"k0 {k}: " + ", ".join(f"{a} {x['ce']}({x['hits']}/{x['n']})" for a, x in arms.items())
                            for k, arms in sorted(d[v]["k0"].items(), key=lambda t: -int(t[0])))
            print(f"   {v:15s} clean {d[v]['clean']} | {row}")
print("\n##### budget eps 16 (k0 28)")
for b, d in S["budget_e16"].items():
    L = d["ladder"] or {"iters": [], "stopped": None}
    last = L["iters"][-1] if L["iters"] else None
    print(f"  b={b:3d} {d['host']:9s} alice {d['alice']} GM {d['gm']['ce']}({d['gm']['hits']}) ladder "
          f"{(str(last['ce']) + '(' + str(last['hits']) + ')') if last else '-'} after {len(L['iters'])} it, stopped={L['stopped']}")
