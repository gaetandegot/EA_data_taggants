"""CPU tests of availability poisoning (run_phi_seed keys=avail, the objective sign, stage aeval)."""
import argparse
import json

import torch

import keysel.data as kd
from keysel import run_phi_seed
from keysel.models import build_cifar_model
from keysel.phi import _load
from keysel.phi_dyn import keys_report
from keysel.phi_recenter import run_recenter
from keysel.scores import key_grad, model_params
from tests.test_phi_seed import fake_data


def test_key_grad_per_image_labels_is_mean_of_single():
    torch.manual_seed(0)
    model = build_cifar_model("tinysmooth_gn")
    X, y = torch.rand(7, 3, 32, 32), torch.randint(0, 100, (7,))
    params = model_params(model)
    g_batch = key_grad(model, X, y, params, chunk=3)
    singles = [key_grad(model, X[i], int(y[i]), params) for i in range(7)]
    for j, g in enumerate(g_batch):
        assert torch.allclose(g, sum(s[j] for s in singles) / 7, atol=1e-6, rtol=1e-4)


def test_keys_report_sign():
    torch.manual_seed(0)
    model = build_cifar_model("tinysmooth_gn")
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    X, y = torch.rand(5, 3, 32, 32), torch.randint(0, 100, (5,))
    a, b = keys_report(model, theta, X, y), keys_report(model, theta, X, y, sign=-1.0)
    assert b["ce"] == -a["ce"] and a["hit"] == b["hit"] and a["margin"] == b["margin"]


def test_avail_split_disjoint_and_deterministic():
    v, a, e = run_phi_seed.avail_split(2, 10000)
    assert len(v) == 1000 and len(a) == 0 and len(e) == 9000 and len(set(v.tolist()) | set(e.tolist())) == 10000
    assert torch.equal(v, run_phi_seed.avail_split(2, 10000)[0]) and not torch.equal(v, run_phi_seed.avail_split(3, 10000)[0])
    v4, a4, e4 = run_phi_seed.avail_split(2, 10000, 4000, 1000)
    assert (len(v4), len(a4), len(e4)) == (4000, 1000, 5000)
    assert len(set(v4.tolist()) | set(a4.tolist()) | set(e4.tolist())) == 10000
    assert torch.equal(v4[:1000], v)  # same permutation: the first runs' 1,000 images are inside the new pool


def test_avail_end_to_end(tmp_path):
    data = fake_data(n_test=3000)
    data_f = tmp_path / "data.pt"
    torch.save(data, data_f)
    seed = 5
    out = tmp_path / "av"
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=1", "per_key=12", "steps=2", "restarts=1", "eps=8",
            "pick=any", "keys=avail", "k0s=2", "ablation=0"]
    cfg = run_phi_seed.config(out, seed, run_phi_seed.parse_set(sets))
    assert run_phi_seed.objective_sign(cfg) == -1.0
    kx, ky, idx, key_of = run_phi_seed.selection(cfg, data, "any")
    vi, _, ei = run_phi_seed.avail_split(seed, len(data[3]))
    assert len(kx) == 1000 and torch.equal(ky, data[3][vi])  # keys = validation images with their TRUE labels
    assert len(idx) == 12 and bool((key_of == 0).all())
    for st in ("alice", "gm", "ablation"):
        run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage=st, data=str(data_f), set=sets))
    P = _load(out / "poisons.pt")
    assert P["obj_sign"] == -1.0 and (P["X_signed"] - data[0][idx].float() / 255).abs().max() <= 8 / 255 + 1e-6
    assert not (out / "ablation" / "results.json").exists()  # ablation skipped for availability
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_f, weights_only=False)
    try:
        run_recenter(out, 2, iters=1, steps=2, ladder=(0.5,), init="gm", device="cpu", log=lambda m: None, eps=8 / 255)
    finally:
        kd.load_cifar100 = real
    h = json.loads((out / "e2" / "k2" / "recenter_gm" / "history.json").read_text())
    assert h["alice_ce"] < 0  # J = minus the validation CE
    run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage="aeval", data=str(data_f), set=sets))
    res = json.loads((out / "aeval.json").read_text())
    assert res["done"] and res["n_val"] == 1000 and res["n_eval"] == 2000 and res["n_poisons"] == 12
    assert {"noise", "gm"} <= set(res["k0"]["2"]) <= {"noise", "gm", "ladder"}
    for r in [res["alice"], *res["k0"]["2"].values()]:
        assert 0 <= r["acc_eval"] <= 1 and 0 <= r["acc_val"] <= 1 and r["ce_val"] > 0
    assert abs(res["alice"]["ce_val"] + h["alice_ce"]) < 1e-4  # same validation loss, opposite sign
    run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage="aeval", data=str(data_f), set=sets))
    assert json.loads((out / "aeval.json").read_text()) == res  # resumable


def test_avail_heldout_acceptance(tmp_path):
    """Option (b): gradient from a 1,500-image pool, acceptance and reports on 500 other images."""
    data = fake_data(n_test=3000)
    data_f = tmp_path / "data.pt"
    torch.save(data, data_f)
    seed = 6
    out = tmp_path / "avho"
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=1", "per_key=12", "steps=2", "restarts=1", "eps=8",
            "pick=any", "keys=avail", "k0s=2", "ablation=0", "avail_val=1500", "avail_acc=500"]
    for st in ("alice", "gm"):
        run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage=st, data=str(data_f), set=sets))
    P = _load(out / "poisons.pt")
    vi, ai, ei = run_phi_seed.avail_split(seed, 3000, 1500, 500)
    assert len(P["keys"][1]) == 1500 and torch.equal(P["keys"][1], data[3][vi])
    assert torch.equal(P["accept_keys"][1], data[3][ai])
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_f, weights_only=False)
    try:
        run_recenter(out, 2, iters=1, steps=2, ladder=(0.5,), init="gm", device="cpu", log=lambda m: None, eps=8 / 255)
    finally:
        kd.load_cifar100 = real
    h = json.loads((out / "e2" / "k2" / "recenter_gm" / "history.json").read_text())
    assert all("ce_grad" in a and len(a["hits"]) == 500 for a in h["attempts"])  # reports on the acceptance set
    run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage="aeval", data=str(data_f), set=sets))
    res = json.loads((out / "aeval.json").read_text())
    assert (res["n_val"], res["n_acc"], res["n_eval"]) == (1500, 500, 1000)
    assert abs(res["alice"]["ce_acc"] + h["alice_ce"]) < 1e-4  # the ladder's J is minus the acceptance-set CE
    assert all("ce_acc" in r and 0 <= r["acc_acc"] <= 1 for r in res["k0"]["2"].values())
