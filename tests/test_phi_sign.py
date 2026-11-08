"""CPU tests of the trajectory-anchored signer (keysel.phi_sign, keysel.run_phi_sign)."""
import json

import pytest
import torch

from keysel.models import build_cifar_model
from keysel.phi import flat, set_flat, train_traj
from keysel.phi_sign import craft_traj, epoch_weights, load_bank
from keysel.scores import head_mask, key_grad, model_params
from keysel.craft import score_of
from tests.test_phi import ARCH, BS, DATA, PIDX, _driver_args, fake_balanced

F32 = torch.float32


@pytest.fixture(scope="module")
def alice(tmp_path_factory):
    d = tmp_path_factory.mktemp("alice")
    train_traj(ARCH, 0, d, DATA, PIDX, epochs=4, bs=BS, lr=0.05, warmup_epochs=0, device="cpu", dtype=F32,
               log=lambda *_: None)
    return d


def setup(alice):
    bank, lrs, th_final = load_bank(alice)
    model = build_cifar_model(ARCH)
    X = DATA[0][PIDX[:4]].float() / 255
    kx = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(5))
    return model, bank, lrs, th_final, X, kx


def test_load_bank_matches_files(alice):
    bank, lrs, th = load_bank(alice)
    assert bank.shape[0] == 4 and len(lrs) == 4 and th.shape == (bank.shape[1],)
    from keysel.phi_metrics import load_pre
    assert torch.equal(bank[2], load_pre(alice, 2)["theta"]) and lrs[2] == load_pre(alice, 2)["lr"]


def test_epoch_weights():
    lrs = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.0], dtype=torch.float64)
    u = epoch_weights(lrs, "uniform")
    assert u.sum() == pytest.approx(1) and torch.allclose(u, torch.full((6,), 1 / 6, dtype=torch.float64))
    late = epoch_weights(lrs, "uniform", e_min=3)
    assert late[:3].sum() == 0 and late[3:].sum() == pytest.approx(1)
    g = epoch_weights(lrs, "gamma")
    assert g[0] == 0 and g[5] == 0 and g[4] / g[1] == pytest.approx(4.0)
    assert epoch_weights(lrs, "uniform", e_min=1, e_max=3).nonzero().flatten().tolist() == [1, 2]
    with pytest.raises(ValueError):
        epoch_weights(lrs, "bogus")
    with pytest.raises(ValueError):
        epoch_weights(lrs, "uniform", e_min=6)


@pytest.mark.parametrize("mode", ["at_e", "final"])
def test_craft_respects_budget_and_is_deterministic(alice, mode):
    model, bank, lrs, th, X, kx = setup(alice)
    p = epoch_weights(lrs)
    kw = dict(key_mode=mode, eps=8 / 255, steps=6, restarts=2, chunk=2)
    a = craft_traj(model, bank, th, kx, 7, X, p, gen=torch.Generator().manual_seed(1), **kw)
    b = craft_traj(model, bank, th, kx, 7, X, p, gen=torch.Generator().manual_seed(1), **kw)
    c = craft_traj(model, bank, th, kx, 7, X, p, gen=torch.Generator().manual_seed(2), **kw)
    d = a["delta"]
    assert d.abs().max() <= 8 / 255 + 1e-7 and ((X + d) >= -1e-7).all() and ((X + d) <= 1 + 1e-7).all()
    assert torch.equal(d, b["delta"]) and not torch.equal(d, c["delta"])
    assert len(a["scores"]) == 2 and a["score"] == max(a["scores"]) and -1 <= a["score"] <= 1


@pytest.mark.parametrize("mode", ["at_e", "final"])
def test_crafting_improves_alignment_over_the_trajectory(alice, mode):
    model, bank, lrs, th, X, kx = setup(alice)
    params = model_params(model)
    mask = head_mask(model, params)

    def score(delta):
        vals = []
        for e in range(bank.shape[0]):
            set_flat(model, th if mode == "final" else bank[e])
            gk = key_grad(model, kx, 7, params)
            set_flat(model, bank[e])
            vals.append(score_of(model, params, mask, gk, X + delta, 7, 2)[0])
        return sum(vals) / len(vals)

    g = torch.Generator().manual_seed(0)
    before = score((torch.randn(X.shape, generator=g) * 0.05).clamp(-8 / 255, 8 / 255))
    r = craft_traj(model, bank, th, kx, 7, X, epoch_weights(lrs), mode, eps=8 / 255, steps=40, restarts=1, chunk=2,
                   gen=torch.Generator().manual_seed(0))
    assert score(r["delta"]) > before + 0.01, (before, score(r["delta"]))


def test_single_epoch_bank_at_e_equals_final(alice):
    model, bank, lrs, th, X, kx = setup(alice)
    one = th[None].clone()  # one checkpoint that IS the final parameters
    p = torch.ones(1, dtype=torch.float64)
    kw = dict(eps=8 / 255, steps=5, restarts=1, chunk=2)
    a = craft_traj(model, one, th, kx, 7, X, p, "at_e", gen=torch.Generator().manual_seed(3), **kw)
    b = craft_traj(model, one, th, kx, 7, X, p, "final", gen=torch.Generator().manual_seed(3), **kw)
    assert torch.equal(a["delta"], b["delta"])


def test_only_the_epoch_support_is_used(alice):
    model, bank, lrs, th, X, kx = setup(alice)
    poisoned = bank.clone()
    poisoned[:2] = float("nan")  # any use of epochs 0,1 would give NaN gradients
    p = epoch_weights(lrs, "uniform", e_min=2)
    r = craft_traj(model, poisoned, th, kx, 7, X, p, "at_e", eps=8 / 255, steps=6, restarts=1, chunk=2,
                   gen=torch.Generator().manual_seed(0))
    assert torch.isfinite(r["delta"]).all() and torch.isfinite(torch.tensor(r["score"]))


def test_crafting_does_not_change_the_bank(alice):
    model, bank, lrs, th, X, kx = setup(alice)
    before = bank.clone()
    craft_traj(model, bank, th, kx, 7, X, epoch_weights(lrs), "at_e", eps=8 / 255, steps=3, restarts=1, chunk=2)
    assert torch.equal(before, bank)


def test_run_phi_sign_end_to_end(tmp_path, monkeypatch):
    import keysel.run_phi as rp
    import keysel.run_phi_sign as rs
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    monkeypatch.setattr(rs, "load_cifar100", lambda device="cpu": fake_balanced())
    rp.main(_driver_args(tmp_path / "o"))
    a = type("A", (), dict(run=str(tmp_path / "o"), variants="all_at_e,late_final", seeds="0,1", steps=3, restarts=1))()
    res = rs.main(a)
    expected = {"alice_s0", "baseline_s0", "alice_s1", "baseline_s1", "all_at_e_s0", "all_at_e_s1",
                "late_final_s0", "late_final_s1"}
    assert set(res) == expected
    for r in res.values():
        assert len(r["hit"]) == 2 and len(r["margin"]) == 2 and r["test_acc"] is not None
    assert (tmp_path / "o" / "signer" / "summary.json").exists()
    assert json.loads((tmp_path / "o" / "signer" / "summary.json").read_text()).keys() == res.keys()
    # re-running reuses crafted poisons and trained models
    res2 = rs.main(a)
    assert res2["all_at_e_s0"]["margin"] == res["all_at_e_s0"]["margin"]
    X1 = torch.load(tmp_path / "o" / "signer" / "all_at_e" / "poisons.pt", weights_only=False)["X_signed"]
    base = torch.load(tmp_path / "o" / "poisons.pt", weights_only=False)["X_signed"]
    assert X1.shape == base.shape and not torch.equal(X1, base)
