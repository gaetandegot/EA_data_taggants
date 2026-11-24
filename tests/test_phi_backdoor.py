"""CPU tests of the backdoor setting (keysel.backdoor, run_phi_seed keys=backdoor, stage beval)."""
import argparse
import json

import torch

import keysel.data as kd
from keysel import run_phi_seed
from keysel.backdoor import apply_trigger, backdoor_spec, triggered_sources
from keysel.models import build_cifar_model
from keysel.phi_recenter import run_recenter
from keysel.scores import key_grad, model_params
from tests.test_phi_seed import fake_data


def test_trigger_and_spec():
    s, t, patch = backdoor_spec(3)
    assert s != t and patch.shape == (3, 8, 8) and backdoor_spec(3)[0] == s
    X = torch.zeros(5, 3, 32, 32)
    Y = apply_trigger(X, patch, seed=1)
    assert torch.equal(X, torch.zeros_like(X))  # input untouched
    for i in range(5):
        nz = Y[i].abs().sum(0).nonzero()
        r, c = int(nz[:, 0].min()), int(nz[:, 1].min())
        assert torch.allclose(Y[i, :, r:r + 8, c:c + 8], patch)
        assert int((Y[i] != 0).sum()) == int((patch != 0).sum())  # nothing else changed
    assert torch.equal(Y, apply_trigger(X, patch, seed=1))


def test_key_grad_batch_is_mean_of_single():
    torch.manual_seed(0)
    model = build_cifar_model("tinysmooth_gn")
    X = torch.rand(7, 3, 32, 32)
    params = model_params(model)
    g_batch = key_grad(model, X, 5, params, chunk=3)
    singles = [key_grad(model, X[i], 5, params) for i in range(7)]
    for j, g in enumerate(g_batch):
        mean = sum(s[j] for s in singles) / 7
        assert torch.allclose(g, mean, atol=1e-6, rtol=1e-4)


def test_backdoor_end_to_end(tmp_path):
    data = fake_data(n_test=3000)
    data_f = tmp_path / "data.pt"
    torch.save(data, data_f)
    seed = 4
    s, t, _ = backdoor_spec(seed)
    out = tmp_path / "bd"
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=1", "per_key=8", "steps=2", "restarts=1", "eps=16",
            "pick=target", "keys=backdoor", "k0s=2"]
    cfg = run_phi_seed.config(out, seed, run_phi_seed.parse_set(sets))
    kx, ky, idx, key_of = run_phi_seed.selection(cfg, data, "target")
    assert len(kx) == int((data[1] == s).sum()) and bool((ky == t).all())
    assert len(idx) == 8 and bool((data[1][idx] == t).all()) and bool((key_of == 0).all())
    Xq, yq = triggered_sources(data, seed, "test")
    assert len(Xq) == int((data[3] == s).sum()) > 0
    for st in ("alice", "gm"):
        run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage=st, data=str(data_f), set=sets))
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_f, weights_only=False)
    try:
        run_recenter(out, 2, iters=1, steps=2, ladder=(0.5,), init="gm", device="cpu", log=lambda m: None, eps=16 / 255)
    finally:
        kd.load_cifar100 = real
    run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage="beval", data=str(data_f), set=sets))
    res = json.loads((out / "beval.json").read_text())
    assert res["done"] and res["source"] == s and res["target"] == t and res["n_poisons"] == 8
    assert "alice" in res and set(res["k0"]["2"]) <= {"gm", "ladder"} and "gm" in res["k0"]["2"]
    for r in [res["alice"], *res["k0"]["2"].values()]:
        assert 0 <= r["asr_test"] <= 1 and 0 <= r["asr_craft"] <= 1 and 0 <= r["clean_acc"] <= 1
    run_phi_seed.main(argparse.Namespace(out=str(out), seed=seed, stage="beval", data=str(data_f), set=sets))
    assert json.loads((out / "beval.json").read_text()) == res  # resumable
