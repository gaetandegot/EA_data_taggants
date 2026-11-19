"""CPU tests of the learning-rate choice, the robustness runs and the data-budget experiment (synthetic data)."""
import argparse
import json

import torch

from keysel import phi_budget, phi_lr_choice, phi_robust, run_phi_seed
from tests.test_phi_seed import fake_data


def _run(tmp_path, name, seed, sets, data_f, stages=("alice", "gm")):
    out = tmp_path / name
    for st in stages:
        run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage=st, data=str(data_f), set=sets))
    return out


def test_lr_choice_rule(tmp_path):
    def fake(lr, losses):
        r = tmp_path / f"lr{lr}"
        (r / "R0").mkdir(parents=True)
        (r / "config.json").write_text(json.dumps({"lr": lr, "epochs": len(losses)}))
        (r / "R0" / "log.jsonl").write_text("\n".join(json.dumps({"epoch": i + 1, "train_loss": x, "test_acc": 0.3})
                                                     for i, x in enumerate(losses)))
        return str(r)
    runs = [fake(0.1, [4.0, 1.5, 0.9]), fake(0.25, [4.0, 1.0, 0.4]), fake(0.5, [4.0, 0.5, 0.2]),
            fake(1.0, [4.0, float("nan"), 9.0])]
    out = tmp_path / "choice.json"
    phi_lr_choice.main(argparse.Namespace(runs=runs, out=str(out), target=0.5))
    c = json.loads(out.read_text())
    assert c["lr"] == 0.25  # smallest lr under the target; the diverged 1.0 is discarded
    phi_lr_choice.main(argparse.Namespace(runs=runs[:1], out=str(out), target=0.5))
    assert json.loads(out.read_text())["lr"] == 0.1  # none reaches the target: lowest loss among the stable ones


def test_robust_export_and_run(tmp_path):
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=10", "per_key=3", "steps=2", "restarts=1", "eps=8",
            "pick=target", "k0s=2"]
    run = _run(tmp_path, "hard", 6, sets, data_f)
    from keysel.phi_recenter import run_recenter
    import keysel.data as kd
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_f, weights_only=False)
    try:
        run_recenter(run, 2, iters=1, steps=2, ladder=(0.5,), init="gm", device="cpu", log=lambda s: None, eps=8 / 255)
    finally:
        kd.load_cifar100 = real
    shared = tmp_path / "shared" / "hard"
    phi_robust.main(argparse.Namespace(export=True, run=str(run), to=str(shared)))
    assert json.loads((shared / "done").read_text())["k0s"] == [2]
    lr_f = tmp_path / "lr.json"
    lr_f.write_text(json.dumps({"lr": 0.05}))
    phi_robust.DEIT = "tinycnn_gn"  # stand-in for DeiT-tiny on CPU
    out = tmp_path / "robust"
    args = argparse.Namespace(export=False, src=str(shared), out=str(out), variants="shuffle,arch,both",
                              lr_file=str(lr_f), data=str(data_f), poisons=None)
    phi_robust.main(args)
    res = json.loads((out / "results.json").read_text())
    assert res["done"] and set(res) >= {"shuffle", "arch", "both"}
    assert res["shuffle"]["spec"]["order_seed"] == 1006 and res["arch"]["spec"]["arch"] == "tinycnn_gn"
    for v in ("shuffle", "arch", "both"):
        assert set(res[v]["k0"]["2"]) == {"gm", "ladder"} and "clean" in res[v]
    phi_robust.main(args)  # resumable: nothing left to do, same results
    assert json.loads((out / "results.json").read_text())["shuffle"] == res["shuffle"]


def test_budget_gm_and_ladder(tmp_path):
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=10", "per_key=4", "steps=2", "restarts=1", "eps=8",
            "pick=target", "k0s=2"]
    run = _run(tmp_path, "budget", 7, sets, data_f)
    phi_budget.main(argparse.Namespace(run=str(run), k0=2, budgets="1,4", summary=False, data=str(data_f)))
    res = json.loads((run / "budget" / "results.json").read_text())
    assert res["done"] and set(res["gm"]) == {"1", "4"}
    P = torch.load(run / "poisons.pt", weights_only=False)
    Xc = fake_data()[0][P["idx"]].float() / 255
    X1 = torch.load(run / "budget" / "poisons_b1.pt", weights_only=False)["X_signed"]
    changed = (X1 - Xc).flatten(1).abs().amax(1) > 0
    for k in range(10):  # only the first image of each key is perturbed at budget 1
        g = (P["key_of"] == k).nonzero().flatten()
        assert changed[g[0]] and not changed[g[1:]].any()
    from keysel.phi_recenter import run_recenter
    import keysel.data as kd
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_f, weights_only=False)
    try:
        h = run_recenter(run, 2, iters=1, steps=2, ladder=(0.5,), device="cpu", log=lambda s: None, eps=8 / 255,
                         init_file=str(run / "budget" / "poisons_b1.pt"), budget=1, subdir="budget_b1")
    finally:
        kd.load_cifar100 = real
    assert len(h["attempts"]) == 1
    cur = run / "e2" / "k2" / "budget_b1" / "poisons_current.pt"
    if cur.exists():  # an accepted iteration may only have moved the budgeted images
        moved = (torch.load(cur, weights_only=False)["X_signed"] - Xc).flatten(1).abs().amax(1) > 0
        assert not (moved & ~changed).any()


def test_craft_B_free_mask(tmp_path):
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=10", "per_key=2", "steps=2", "restarts=1", "eps=8",
            "pick=target", "k0s=2"]
    run = _run(tmp_path, "mask", 8, sets, data_f)
    rp, P, cfg = phi_budget._replay_with(run, "cpu", data_f)
    from keysel.phi_dyn import craft_B
    th = {e: rp.pre(e) for e in (2,)}
    Pe = {2: torch.randn_like(th[2]) * 1e-2}
    X, y = rp.Xf[P["idx"]], rp.ytr[P["idx"]]
    free = torch.zeros(len(X), dtype=torch.bool)
    free[::2] = True
    r = craft_B(rp, Pe, 2, X, y, [torch.arange(len(X))], 8 / 255, steps=3, restarts=1, thetas=th, free=free,
                log=lambda s: None)
    d = r["delta"].flatten(1).abs().amax(1)
    assert (d[~free] == 0).all() and (d[free] > 0).all() and (d <= 8 / 255 + 1e-7).all()


def test_robust_reverse_direction_from_chosen_folder(tmp_path, monkeypatch):
    """DeiT-crafted poisons (exported from <job>/chosen) -> Bobs of the other architecture (ResNet at lr 0.05)."""
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    monkeypatch.setattr(phi_robust, "DEIT", "tinycnn_gn")        # stand-in for DeiT-tiny
    monkeypatch.setattr(phi_robust, "RESNET", "tinysmooth_gn")   # stand-in for ResNet18-GN-SiLU
    job = tmp_path / "deit_auto_s4"
    sets = ["arch=tinycnn_gn", "epochs=3", "n_keys=10", "per_key=2", "steps=2", "restarts=1", "eps=8", "pick=target",
            "k0s=2", "lr=0.25"]
    run = _run(job, "lr0.25", 4, sets, data_f)
    (job / "chosen").symlink_to("lr0.25")
    shared = tmp_path / "shared" / job.name
    phi_robust.main(argparse.Namespace(export=True, run=str((job / "chosen").resolve()), to=str(shared)))
    assert json.loads((shared / "done").read_text())["ladder_is_gm"] == []  # no ladder yet: no k0 exported
    # with one ladder history the export falls back to the GM poisons for it
    (run / "e2" / "k2" / "recenter_gm").mkdir(parents=True)
    (run / "e2" / "k2" / "recenter_gm" / "history.json").write_text(json.dumps({"accepted": []}))
    phi_robust.main(argparse.Namespace(export=True, run=str((job / "chosen").resolve()), to=str(shared)))
    assert json.loads((shared / "done").read_text()) ["ladder_is_gm"] == [2]
    out = tmp_path / "robust_deit"
    phi_robust.main(argparse.Namespace(export=False, src=str(shared), out=str(out), variants="shuffle,arch,both",
                                       lr_file=None, data=str(data_f)))
    res = json.loads((out / "results.json").read_text())
    assert res["shuffle"]["spec"] == {"arch": "tinycnn_gn", "order_seed": 1004, "init_seed": 4, "lr": 0.25}
    assert res["arch"]["spec"] == {"arch": "tinysmooth_gn", "order_seed": 4, "init_seed": 4, "lr": 0.05}
    assert res["both"]["spec"]["arch"] == "tinysmooth_gn" and res["both"]["spec"]["order_seed"] == 1004
    assert res["done"]


def test_robust_spread_placement(tmp_path):
    """shuffle_spread: GM only, from k0 and epoch 0, plain shuffle (no poison batch); with the clean images as
    'poisons', the run from epoch 0 must reproduce the clean spread training exactly."""
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=10", "per_key=3", "steps=2", "restarts=1", "eps=8",
            "pick=target", "k0s=2"]
    run = _run(tmp_path, "hard", 6, sets, data_f)
    P = torch.load(run / "poisons.pt", weights_only=False)
    Xtr = torch.load(data_f, weights_only=False)[0]
    Xc = Xtr[P["idx"]].float() / 255 if Xtr.dtype == torch.uint8 else Xtr[P["idx"]].clone()
    for name, X in (("gm", P["X_signed"]), ("noop", Xc)):
        d = tmp_path / f"src_{name}"
        d.mkdir()
        (d / "config.json").write_text((run / "config.json").read_text())
        torch.save({**P, "X_signed": X}, d / "poisons.pt")
        torch.save({"X_signed": X}, d / "ladder_k2.pt")
        (d / "done").write_text(json.dumps({"k0s": [2], "ladder_is_gm": []}))
    res = {}
    for name in ("gm", "noop"):
        out = tmp_path / f"spread_{name}"
        phi_robust.main(argparse.Namespace(export=False, src=str(tmp_path / f"src_{name}"), out=str(out),
                                           variants="shuffle_spread", lr_file=None, data=str(data_f), poisons=None))
        res[name] = json.loads((out / "results.json").read_text())["shuffle_spread"]
    r = res["gm"]
    assert r["spec"]["placement"] == "spread" and r["spec"]["order_seed"] == 1006
    assert set(r["k0"]) == {"0", "2"} and all(set(row) == {"gm"} for row in r["k0"].values())
    noop = res["noop"]
    assert noop["k0"]["0"]["gm"]["ce"] == noop["clean"]["ce"]  # clean images in place = the clean training
    assert r["k0"]["0"]["gm"]["ce"] != r["clean"]["ce"]


def test_robust_gm_only_from_run_poisons_file(tmp_path):
    """--poisons: a poisons file read directly from a local run folder (no export), GM only, both placements."""
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=10", "per_key=3", "steps=2", "restarts=1", "eps=8", "k0s=2"]
    run = _run(tmp_path, "lf", 6, sets, data_f)
    out = tmp_path / "same"
    phi_robust.main(argparse.Namespace(export=False, src=str(run), out=str(out), variants="shuffle,shuffle_spread",
                                       lr_file=None, data=str(data_f), poisons="poisons.pt"))
    res = json.loads((out / "results.json").read_text())
    assert res["done"] and set(res["shuffle"]["k0"]) == {"2"} and set(res["shuffle_spread"]["k0"]) == {"0", "2"}
    assert all(set(row) == {"gm"} for v in ("shuffle", "shuffle_spread") for row in res[v]["k0"].values())

