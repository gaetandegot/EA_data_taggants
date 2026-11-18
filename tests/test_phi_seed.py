"""CPU smoke test of the multi-seed driver (keysel.run_phi_seed) and of label-free poisons.

Tiny synthetic CIFAR-shaped data (100 classes x 60 images), tinysmooth_gn, 3 epochs: alice -> gm (any label) ->
ablation (same label) -> one ladder iteration of keysel.phi_recenter from --init gm.
"""
import argparse
import json

import torch

from keysel.craft import craft_key
from keysel.data import poison_subsets, signing_subset
from keysel.models import build_cifar_model
from keysel import run_phi_seed


def fake_data(n_per=60, nc=100, n_test=200):
    g = torch.Generator().manual_seed(0)
    ytr = torch.arange(nc).repeat_interleave(n_per)
    Xtr = torch.randint(0, 256, (len(ytr), 3, 32, 32), generator=g, dtype=torch.uint8)
    yte = torch.randint(0, nc, (n_test,), generator=g)
    Xte = torch.randint(0, 256, (n_test, 3, 32, 32), generator=g, dtype=torch.uint8)
    return Xtr, ytr, Xte, yte


def test_poison_subsets_any_and_target():
    ytr = torch.arange(100).repeat_interleave(60)
    ky = torch.tensor([3, 7, 3, 9])  # two keys share a target class
    anyp = poison_subsets(ytr, ky, 20, seed=1, pick="any")
    flat = torch.cat(anyp)
    assert len(set(flat.tolist())) == 80
    assert (ytr[flat] != 3).any()  # not restricted to the targets
    tgt = poison_subsets(ytr, ky, 20, seed=1, pick="target")
    assert len(set(torch.cat(tgt).tolist())) == 80  # disjoint even for the shared class
    for k, c in enumerate(ky.tolist()):
        assert (ytr[tgt[k]] == c).all()
    assert torch.equal(tgt[1], signing_subset(ytr, 7, 20, 1))  # same images as the old rule for a unique class
    assert torch.equal(poison_subsets(ytr, ky, 20, 1, "any")[2], anyp[2])  # deterministic


def test_craft_key_tensor_labels_match_scalar():
    torch.manual_seed(0)
    m = build_cifar_model("tinysmooth_gn", 10).eval()
    X = torch.rand(6, 3, 32, 32)
    kx = torch.rand(3, 32, 32)
    kw = dict(eps=8 / 255, steps=3, restarts=1, augment=False, chunk=4)
    a = craft_key(m, kx, 2, X, gen=torch.Generator().manual_seed(1), **kw)
    b = craft_key(m, kx, 2, X, gen=torch.Generator().manual_seed(1), y_poison=torch.full((6,), 2), **kw)
    assert torch.equal(a["delta"], b["delta"])
    c = craft_key(m, kx, 2, X, gen=torch.Generator().manual_seed(1), y_poison=torch.tensor([0, 1, 2, 3, 4, 5]), **kw)
    assert not torch.equal(a["delta"], c["delta"])


def test_run_phi_seed_end_to_end(tmp_path):
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    out = tmp_path / "run"
    out.mkdir()
    cfg = dict(run_phi_seed.DEFAULTS, arch="tinysmooth_gn", epochs=3, n_keys=10, per_key=5, steps=2, restarts=1,
               k0s=[2], out=str(out), seed=4)
    (out / "config.json").write_text(json.dumps(cfg))
    for stage in ("alice", "gm", "ablation"):
        run_phi_seed.main(argparse.Namespace(out=str(out), seed=4, stage=stage, data=str(data_f)))
    P = torch.load(out / "poisons.pt", weights_only=False)
    ytr = fake_data()[1]
    _, ky = P["keys"]
    assert len(P["idx"]) == 50 and P["pick"] == "any"
    assert (ytr[P["idx"]] != ky[P["key_of"]]).float().mean() > 0.5  # label-free
    S = torch.load(out / "ablation" / "poisons_same.pt", weights_only=False)
    assert (ytr[S["idx"]] == ky[S["key_of"]]).all()
    res = json.loads((out / "ablation" / "results.json").read_text())
    assert res["done"] and set(res["k0"]["2"]) == {"gm_any", "gm_same", "gm_same_last"} and res["order"] == "alice"
    assert len(res["k0"]["2"]["gm_any"]["hit"]) == 10
    # gm_any through bob_values (poisons at their own positions = the last batch) equals the ladder's bob_traj
    from keysel.phi_recenter import bob_traj
    from keysel.phi_dyn import keys_report
    cfgd = json.loads((out / "config.json").read_text())
    data = torch.load(data_f, weights_only=False)
    rp = run_phi_seed.make_replay(out, cfgd, data, P["idx"], "cpu")
    th_ladder = bob_traj(rp, 2, P["X_signed"])[0]
    th_ab = run_phi_seed.bob_values(rp, rp.start(1), 2, run_phi_seed.with_poisons(rp, P["idx"], P["X_signed"]))
    assert torch.allclose(th_ladder, th_ab, atol=1e-6)
    alice_th = run_phi_seed.bob_values(rp, rp.start(1), 2, rp.Xf)
    assert torch.allclose(alice_th, torch.load(out / "R0" / "final.pt", weights_only=False)["theta"], atol=1e-5)

    from keysel.phi_recenter import run_recenter
    import keysel.data as kd
    real = kd.load_cifar100
    kd.load_cifar100 = lambda device="cpu", **k: torch.load(data_f, weights_only=False)
    try:
        hist = run_recenter(out, 2, iters=1, steps=2, ladder=(0.5,), init="gm", device="cpu", cache=True,
                            log=lambda s: None)
    finally:
        kd.load_cifar100 = real
    assert len(hist["attempts"]) == 1 and len(hist["attempts"][0]["hits"]) == 10


def test_bob_swapped_order(tmp_path):
    """Swapped order: S_new last, each image once per epoch, only the swapped positions differ from Alice's order."""
    data_f = tmp_path / "data.pt"
    data = fake_data()
    out = tmp_path / "run"
    out.mkdir()
    cfg = dict(run_phi_seed.DEFAULTS, arch="tinysmooth_gn", epochs=3, n_keys=10, per_key=5, out=str(out), seed=4)
    S = torch.cat(poison_subsets(data[1], torch.arange(10), 5, 4, "any"))
    tgt = [i for i in torch.cat(poison_subsets(data[1], torch.arange(10), 5, 4, "target")).tolist() if i not in S[:3].tolist()]
    S_new = torch.tensor(S[:3].tolist() + tgt[:len(S) - 3])  # same size, 3 images shared with S
    assert len(set(S_new.tolist())) == len(S)
    rp = run_phi_seed.make_replay(out, cfg, data, S, "cpu")
    seen = []
    rp.sgd = lambda th, x, y, lr: (seen.append(x), th)[1]  # record the batches only
    X_all = torch.arange(len(rp.Xf), dtype=torch.float32)[:, None, None, None].expand(-1, 1, 1, 1)  # pixel = index
    rp.Xf = X_all
    run_phi_seed.bob_swapped(rp, torch.zeros(1), 2, X_all, S_new)
    swapped = [x.flatten().long() for x in seen]
    seen.clear()
    run_phi_seed.bob_values(rp, torch.zeros(1), 2, X_all)
    alice = [x.flatten().long() for x in seen]
    assert len(swapped) == len(alice)
    assert torch.equal(swapped[-1], S_new)
    assert sorted(torch.cat(swapped).tolist()) == list(range(len(rp.Xf)))  # one epoch: every image exactly once
    moved = set(S.tolist()) ^ set(S_new.tolist())
    for a, b in zip(alice[:-1], swapped[:-1]):
        diff = (a != b)
        assert set(a[diff].tolist()) <= moved and set(b[diff].tolist()) <= moved
    seen.clear()
    run_phi_seed.bob_swapped(rp, torch.zeros(1), 2, X_all, S)  # no swap = Alice's order
    assert all(torch.equal(x.flatten().long(), a) for x, a in zip(seen, alice))


def test_config_overrides_and_select(tmp_path):
    """--set overrides (eps / keys / pick, same-label run skips the ablation) and keysel.phi_select end to end."""
    data_f = tmp_path / "data.pt"
    torch.save(fake_data(), data_f)
    assert run_phi_seed.parse_set(["n_keys=20", "eps=8", "pick=target", "ablation=0", "k0s=2"]) == \
        {"n_keys": 20, "eps": 8.0, "pick": "target", "ablation": False, "k0s": [2]}
    # same-label run created through --set: config written once, ablation skipped
    out_t = tmp_path / "hard"
    sets = ["arch=tinysmooth_gn", "epochs=3", "n_keys=20", "per_key=2", "steps=2", "restarts=1", "eps=8", "pick=target",
            "k0s=2"]
    for stage in ("alice", "gm", "ablation"):
        run_phi_seed.main(argparse.Namespace(out=str(out_t), seed=5, stage=stage, data=str(data_f), set=sets))
    cfg = json.loads((out_t / "config.json").read_text())
    assert cfg["eps"] == 8.0 and cfg["n_keys"] == 20 and cfg["pick"] == "target"
    assert not (out_t / "ablation" / "results.json").exists()
    P = torch.load(out_t / "poisons.pt", weights_only=False)
    assert (fake_data()[1][P["idx"]] == P["keys"][1][P["key_of"]]).all()
    assert float((P["X_signed"] - fake_data()[0][P["idx"]].float() / 255).abs().max()) <= 8 / 255 + 1e-6
    import pytest
    with pytest.raises(AssertionError):  # an existing run folder cannot be re-created with other settings
        run_phi_seed.main(argparse.Namespace(out=str(out_t), seed=5, stage="gm", data=str(data_f), set=["eps=16"]))
    # selection on a label-free run
    out = tmp_path / "run"
    out.mkdir()
    cfg = dict(run_phi_seed.DEFAULTS, arch="tinysmooth_gn", epochs=3, n_keys=10, per_key=5, steps=2, restarts=1,
               k0s=[2], out=str(out), seed=4)
    (out / "config.json").write_text(json.dumps(cfg))
    for stage in ("alice", "gm", "ablation"):
        run_phi_seed.main(argparse.Namespace(out=str(out), seed=4, stage=stage, data=str(data_f)))
    from keysel import phi_select
    phi_select.main(argparse.Namespace(run=str(out), k0=2, steps=2, restarts=1, data=str(data_f)))
    res = json.loads((out / "select_k2" / "results.json").read_text())
    assert res["done"] and set(res["bob"]) == {"any", "same", "selected"}
    assert res["scores"]["selected"]["sum"] >= max(res["scores"]["any"]["sum"], res["scores"]["same"]["sum"])
    score = torch.load(out / "select_k2" / "scores.pt", weights_only=False)
    assert score.shape == (len(fake_data()[1]),) and (score >= 0).all()


def test_chunked_hvp_equals_full_batch():
    """hvp() splits batches larger than KEYSEL_HVP_CHUNK; the size-weighted sum must equal the full-batch HVP."""
    import keysel.phi_lin as L
    from keysel.phi import flat
    torch.manual_seed(0)
    m = build_cifar_model("tinysmooth_gn").double()
    th = flat(m).clone()
    v = torch.randn_like(th)
    x = torch.rand(10, 3, 32, 32, dtype=torch.float64)
    y = torch.randint(0, 100, (10,))
    old = L.HVP_CHUNK
    try:
        L.HVP_CHUNK = 1000
        full = L.hvp(m, th, x, y, v)
        L.HVP_CHUNK = 4  # chunks of 4, 4, 2
        chunked = L.hvp(m, th, x, y, v)
    finally:
        L.HVP_CHUNK = old
    assert float((chunked - full).norm() / full.norm()) < 1e-12
