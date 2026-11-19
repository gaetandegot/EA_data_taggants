"""CPU tests of the Phi=I measurement (keysel.phi, keysel.phi_metrics). float64 for exact identities."""
import os
import time

import pytest
import torch

from keysel.craft import craft_key
from keysel.models import build_cifar_model
from keysel.phi import (batch_grad, epoch_plan, flat, poison_scale, kick_at, lr_at, random_direction, set_flat,
                        train_traj)
from keysel.phi_metrics import (compute_metrics, drift_curve, key_stats, load_final, load_pre, n_epochs,
                                write_metrics)

D = torch.float64
ARCH = "tinycnn_gn"
BS = 32


def fake(n=100, nt=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)
    y = torch.randint(0, 100, (n,), generator=g)
    Xt = torch.randint(0, 256, (nt, 3, 32, 32), generator=g, dtype=torch.uint8)
    return X, y, Xt, torch.randint(0, 100, (nt,), generator=g)


DATA = fake()
PIDX = torch.tensor([3, 17, 42, 58, 71, 90])


def signed(eps=8 / 255, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    x = DATA[0][PIDX].to(D) / 255
    d = eps * scale * torch.sign(torch.randn(x.shape, generator=g, dtype=D))
    return (x + d).clamp(0, 1)


def run(tmp, name, X_signed=None, seed=0, epochs=3, **kw):
    kw.setdefault("lr", 0.05)
    kw.setdefault("warmup_epochs", 0)
    return train_traj(ARCH, seed, tmp / name, DATA, PIDX, X_signed=X_signed, epochs=epochs, bs=BS, device="cpu",
                      dtype=D, log=lambda *_: None, **kw)


# ----------------------------------------------------------------------------- model
def test_gn_models_have_no_running_stats():
    for arch in ("resnet18_gn", ARCH):
        names = [n for n, _ in build_cifar_model(arch).named_buffers()]
        assert set(names) == {"mean", "std"}, names
    assert any("running_mean" in n for n, _ in build_cifar_model("resnet18").named_buffers())


def test_gn_per_sample_independence_in_train_mode():
    for arch in ("resnet18_gn", ARCH):
        m = build_cifar_model(arch).double().train()
        x = torch.rand(4, 3, 32, 32, dtype=D)
        full = m(x)
        for i in range(4):
            assert torch.allclose(full[i], m(x[i:i + 1])[0], atol=1e-10)
        x2 = x.clone()
        x2[1:] = torch.rand(3, 3, 32, 32, dtype=D)  # change the other images of the batch
        assert torch.allclose(full[0], m(x2)[0], atol=1e-10)


def test_resnet18_gn_param_count_and_train_equals_eval():
    m = build_cifar_model("resnet18_gn")
    assert 11_000_000 < sum(p.numel() for p in m.parameters()) < 11_500_000
    x = torch.rand(2, 3, 32, 32)
    m.train(); a = m(x)
    m.eval(); b = m(x)
    assert torch.equal(a, b)


# ----------------------------------------------------------------------------- schedules and plan
def test_lr_at():
    assert lr_at(0, 100, 0.1, 10, "const") == pytest.approx(0.01)
    assert lr_at(9, 100, 0.1, 10, "const") == pytest.approx(0.1)
    assert lr_at(50, 100, 0.1, 10, "const") == pytest.approx(0.1)
    assert lr_at(10, 100, 0.1, 10, "cosine") == pytest.approx(0.1)
    assert lr_at(100, 100, 0.1, 10, "cosine") == pytest.approx(0.0, abs=1e-12)
    assert lr_at(55, 100, 0.1, 10, "cosine") == pytest.approx(0.05)
    assert lr_at(3, 100, 0.1, 0, "cosine") < 0.1
    vals = [lr_at(s, 100, 0.1, 10, "cosine") for s in range(10, 100)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    with pytest.raises(ValueError):
        lr_at(0, 1, 0.1, 0, "bogus")


def test_epoch_plan_structure():
    n = 100
    plan = epoch_plan(n, PIDX, 0, 0, BS)
    assert torch.equal(plan[-1], PIDX)  # poisons last, own batch
    clean = torch.cat(plan[:-1])
    assert len(clean) == n - len(PIDX) and len(set(clean.tolist())) == len(clean)
    assert not set(clean.tolist()) & set(PIDX.tolist())
    assert all(len(b) == BS for b in plan[:-2]) and 0 < len(plan[-2]) <= BS


def test_epoch_plan_determinism_and_variation():
    a, b = epoch_plan(100, PIDX, 2, 7, BS), epoch_plan(100, PIDX, 2, 7, BS)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    c, d = epoch_plan(100, PIDX, 3, 7, BS), epoch_plan(100, PIDX, 2, 8, BS)
    assert not torch.equal(torch.cat(a[:-1]), torch.cat(c[:-1]))
    assert not torch.equal(torch.cat(a[:-1]), torch.cat(d[:-1]))


def test_epoch_plan_edge_cases():
    assert len(epoch_plan(100, torch.tensor([], dtype=torch.long), 0, 0, BS)) == 4  # no poison batch
    big = epoch_plan(100, torch.arange(BS + 1), 0, 0, BS)  # more poisons than a batch: one merged poison step
    assert len(big[-1]) == BS + 1 and poison_scale(BS + 1, BS) == 2 and poison_scale(BS, BS) == 1
    full = epoch_plan(100, torch.arange(BS), 0, 0, BS)  # exactly one full poison batch
    assert len(full[-1]) == BS


# ----------------------------------------------------------------------------- trajectories
def test_determinism_bitwise(tmp_path):
    a, b = run(tmp_path, "a"), run(tmp_path, "b")
    assert torch.equal(flat(a), flat(b))
    for e in range(3):
        assert torch.equal(load_pre(tmp_path / "a", e)["theta"], load_pre(tmp_path / "b", e)["theta"])


def test_different_seeds_differ(tmp_path):
    assert not torch.equal(flat(run(tmp_path, "a", seed=0)), flat(run(tmp_path, "b", seed=1)))


def test_bob_equals_alice_when_delta_zero(tmp_path):
    x_clean = DATA[0][PIDX].to(D) / 255
    a, b = run(tmp_path, "a"), run(tmp_path, "b", X_signed=x_clean)
    assert torch.equal(flat(a), flat(b))


def test_trajectories_diverge_only_after_first_poison_step(tmp_path):
    run(tmp_path, "a"); run(tmp_path, "b", X_signed=signed())
    assert torch.equal(load_pre(tmp_path / "a", 0)["theta"], load_pre(tmp_path / "b", 0)["theta"])
    for e in (1, 2):
        assert not torch.equal(load_pre(tmp_path / "a", e)["theta"], load_pre(tmp_path / "b", e)["theta"])
    assert drift_curve(tmp_path / "b", tmp_path / "a")[0] == 0.0


def test_pre_checkpoint_is_really_pre_poison(tmp_path):
    """After 1 epoch, final = pre - lr * (grad of the poison batch + wd * pre)."""
    xs, wd = signed(), 5e-4
    m = run(tmp_path, "b", X_signed=xs, epochs=1, wd=wd)
    pre = load_pre(tmp_path / "b", 0)
    y = DATA[1][PIDX]
    g = batch_grad(build_cifar_model(ARCH).to(D), pre["theta"], xs, y)
    expect = pre["theta"] - pre["lr"] * (g + wd * pre["theta"])
    assert torch.allclose(flat(m), expect, atol=1e-12, rtol=0)


def test_pre_files_and_final_saved(tmp_path):
    run(tmp_path, "a", epochs=4)
    assert n_epochs(tmp_path / "a") == 4 and (tmp_path / "a" / "final.pt").exists()
    assert not (tmp_path / "a" / "checkpoint.pt").exists()
    assert len((tmp_path / "a" / "log.jsonl").read_text().splitlines()) == 4


def test_save_pre_false(tmp_path):
    run(tmp_path, "a", save_pre=False)
    assert n_epochs(tmp_path / "a") == 0


def test_rerun_returns_cached_final(tmp_path):
    a = run(tmp_path, "a")
    t = time.time()
    b = run(tmp_path, "a")
    assert torch.equal(flat(a), flat(b)) and time.time() - t < 5


def test_resume_is_bitwise_identical(tmp_path):
    full = run(tmp_path, "full", X_signed=signed(), epochs=3)
    os.environ["GPU_DEADLINE"] = str(time.time() + 0.0)
    try:
        run(tmp_path, "cut", X_signed=signed(), epochs=3)
    except SystemExit as e:
        assert e.code == 75
    finally:
        del os.environ["GPU_DEADLINE"]
    assert (tmp_path / "cut" / "checkpoint.pt").exists() and not (tmp_path / "cut" / "final.pt").exists()
    res = run(tmp_path, "cut", X_signed=signed(), epochs=3)
    assert torch.equal(flat(full), flat(res))
    for e in range(3):
        assert torch.equal(load_pre(tmp_path / "full", e)["theta"], load_pre(tmp_path / "cut", e)["theta"])


def test_init_noise_has_exact_relative_norm(tmp_path):
    base, noisy = run(tmp_path, "a", epochs=0), run(tmp_path, "b", epochs=0, init_noise=1e-6)
    d = (flat(noisy) - flat(base)).norm() / flat(base).norm()
    assert float(d) == pytest.approx(1e-6, rel=1e-9)
    assert torch.equal(flat(run(tmp_path, "c", epochs=0, init_noise=1e-6)), flat(noisy))  # seeded


def test_inject_at_requested_epoch(tmp_path):
    """lr=0 freezes training, so pre_e = theta0 (+ injected vector from the injection epoch on)."""
    vec = random_direction(sum(p.numel() for p in build_cifar_model(ARCH).parameters()), 0.5, 3, dtype=D)
    run(tmp_path, "a", lr=0.0, wd=0.0, sched="const")
    run(tmp_path, "b", lr=0.0, wd=0.0, sched="const", inject=(1, vec))
    a0, b0 = (load_pre(tmp_path / n, 0)["theta"] for n in "ab")
    a1, b1 = (load_pre(tmp_path / n, 1)["theta"] for n in "ab")
    assert torch.equal(a0, b0)
    assert torch.allclose(b1 - a1, vec, atol=1e-12)


def test_random_direction_norm_and_seed():
    v = random_direction(1000, 2.5, 0, dtype=D)
    assert float(v.norm()) == pytest.approx(2.5)
    assert torch.equal(v, random_direction(1000, 2.5, 0, dtype=D))
    assert not torch.equal(v, random_direction(1000, 2.5, 1, dtype=D))


def test_resnet18_gn_one_epoch_runs(tmp_path):
    data = fake(24, 8)
    pidx = torch.tensor([1, 5])
    m = train_traj("resnet18_gn", 0, tmp_path / "r", data, pidx, X_signed=data[0][pidx].float() / 255,
                   epochs=1, bs=8, lr=0.01, device="cpu", log=lambda *_: None)
    assert load_pre(tmp_path / "r", 0)["theta"].numel() == flat(m).numel()
    assert torch.isfinite(flat(m)).all()


# ----------------------------------------------------------------------------- kicks
def test_kick_zero_without_signing_and_continuous_in_delta():
    m = build_cifar_model(ARCH).to(D)
    th = flat(m).clone()
    xc, y = DATA[0][PIDX].to(D) / 255, DATA[1][PIDX]
    assert kick_at(m, th, xc, xc, y).abs().max() == 0
    # (not linear: ReLU kink crossings make the gradient piecewise constant in the pixels)
    d = signed(eps=1.0) - xc
    norms = [float(kick_at(m, th, xc, (xc + s * d).clamp(0, 1), y).norm()) for s in (1e-1, 1e-3, 1e-5, 1e-7)]
    assert norms[0] > norms[1] > norms[2] > norms[3] > 0 and norms[3] < 1e-3 * norms[0]


def test_batch_grad_matches_autograd_and_restores_nothing_else():
    m = build_cifar_model(ARCH).to(D)
    th = flat(m).clone()
    x, y = DATA[0][:8].to(D) / 255, DATA[1][:8]
    g = batch_grad(m, th, x, y)
    set_flat(m, th)
    loss = torch.nn.functional.cross_entropy(m(x), y)
    ref = torch.cat([q.reshape(-1) for q in torch.autograd.grad(loss, list(m.parameters()))])
    assert torch.allclose(g, ref, atol=1e-12)


# ----------------------------------------------------------------------------- metrics
def test_single_epoch_eta_is_exactly_minus_gamma_kick(tmp_path):
    """Poison step is the last step of epoch 1 and both models are identical before it => eta = -gamma*kick."""
    xs = signed()
    run(tmp_path, "a", epochs=1); run(tmp_path, "b", X_signed=xs, epochs=1)
    eta = load_final(tmp_path / "b") - load_final(tmp_path / "a")
    pre = load_pre(tmp_path / "a", 0)
    k = kick_at(build_cifar_model(ARCH).to(D), pre["theta"], DATA[0][PIDX].to(D) / 255, xs, DATA[1][PIDX])
    assert torch.allclose(eta, -pre["lr"] * k, atol=1e-13, rtol=0)
    rows, s = compute_metrics(ARCH, tmp_path / "a", tmp_path / "b", DATA, PIDX, xs, device="cpu", dtype=D)
    assert s["cos_eta_S0"] == pytest.approx(1.0, abs=1e-9) and s["ratio_S0"] == pytest.approx(1.0, abs=1e-9)
    assert s["cos_eta_SB"] == pytest.approx(1.0, abs=1e-9)


def test_metrics_structure_and_epoch0_identities(tmp_path):
    xs = signed()
    run(tmp_path, "a"); run(tmp_path, "b", X_signed=xs)
    kx = torch.rand(3, 3, 32, 32, dtype=D)
    ky = torch.tensor([4, 8, 15])
    rows, s = compute_metrics(ARCH, tmp_path / "a", tmp_path / "b", DATA, PIDX, xs, keys=(kx, ky),
                              device="cpu", dtype=D)
    assert len(rows) == 4 and [r["epoch"] for r in rows] == [0, 1, 2, 3]
    r0 = rows[0]
    assert r0["drift"] == 0 and r0["cos_anchor"] == pytest.approx(1.0) and r0["norm_ratio_anchor"] == pytest.approx(1.0)
    assert "cos_eta_S0" not in r0 and "cos_eta_S0" in rows[1] and "cos_anchor" not in rows[-1]
    assert rows[1]["drift"] > 0 and r0["agree"] == 1.0 and r0["kl"] == pytest.approx(0, abs=1e-12)
    assert len(s["cos_eta_lastk"]) == 3 and s["cos_eta_lastk"][-1] == pytest.approx(s["cos_eta_S0"])
    assert s["epochs"] == 3 and -1 <= s["min_cos_anchor"] <= 1
    for k in ("q_dot_eta", "q_dot_S0", "q_dot_SB", "J_A", "J_B", "key_hit_A", "key_hit_B"):
        assert k in s
    write_metrics(rows, s, tmp_path / "out")
    assert (tmp_path / "out" / "metrics.csv").exists() and (tmp_path / "out" / "summary.json").exists()


def test_cos_eta_S0_is_one_for_last_epoch_only_in_the_limit(tmp_path):
    """lastk[0] (only the final kick) can be < 1 but the full sum S0 must explain eta better than a random vector."""
    xs = signed()
    run(tmp_path, "a", epochs=4); run(tmp_path, "b", X_signed=xs, epochs=4)
    _, s = compute_metrics(ARCH, tmp_path / "a", tmp_path / "b", DATA, PIDX, xs, device="cpu", dtype=D)
    assert s["cos_eta_S0"] > 0.5 and s["cos_eta_SB"] > 0.5


def test_eta0_error_scales_with_learning_rate(tmp_path):
    """Phi=I is a first-order-in-gamma approximation: relative error must shrink ~linearly with gamma."""
    xs, errs = signed(), {}
    for lr in (0.08, 0.02):
        run(tmp_path, f"a{lr}", epochs=4, lr=lr, sched="const", wd=0.0)
        run(tmp_path, f"b{lr}", X_signed=xs, epochs=4, lr=lr, sched="const", wd=0.0)
        _, s = compute_metrics(ARCH, tmp_path / f"a{lr}", tmp_path / f"b{lr}", DATA, PIDX, xs, device="cpu", dtype=D)
        errs[lr] = 1 - s["cos_eta_S0"]
    assert errs[0.02] < 0.5 * errs[0.08], errs


def test_perturbation_growth_curve(tmp_path):
    """R1-style run: a tiny init perturbation; the drift curve starts at ~its size and is finite and positive."""
    run(tmp_path, "a", epochs=3)
    run(tmp_path, "p", epochs=3, init_noise=1e-6)
    c = drift_curve(tmp_path / "p", tmp_path / "a")
    assert len(c) == 4 and c[0] > 0 and all(v == v and v < 1.0 for v in c)


def test_key_stats(tmp_path):
    m = build_cifar_model(ARCH).to(D)
    th = flat(m).clone()
    kx = torch.rand(5, 3, 32, 32, dtype=D)
    ky = torch.tensor([0, 1, 2, 3, 4])
    margin, hit, ce = key_stats(m, th, kx, ky)
    z = m(kx)
    ref = torch.stack([z[i, ky[i]] - torch.cat([z[i, :ky[i]], z[i, ky[i] + 1:]]).max() for i in range(5)]).mean()
    assert margin == pytest.approx(float(ref)) and 0 <= hit <= 1 and ce == pytest.approx(
        float(torch.nn.functional.cross_entropy(z, ky)))


# ----------------------------------------------------------------------------- crafting with GN
def test_craft_key_on_gn_model_respects_budget():
    torch.manual_seed(0)
    m = build_cifar_model(ARCH)
    X = DATA[0][:6].float() / 255
    kx, ky = torch.rand(3, 32, 32), 7
    out = craft_key(m, kx, ky, X, eps=8 / 255, steps=4, restarts=2, augment=False, R=1, chunk=3)
    d = out["delta"]
    assert d.shape == X.shape and d.abs().max() <= 8 / 255 + 1e-7
    assert ((X + d) >= -1e-7).all() and ((X + d) <= 1 + 1e-7).all()
    assert -1.0 <= out["full"] <= 1.0


# ----------------------------------------------------------------------------- driver end to end
def fake_balanced(n=600, nt=40):
    X, y, Xt, yt = fake(n, nt)
    return X, torch.arange(n) % 100, Xt, yt  # 6 images per class


def _driver_args(out, epochs=3):
    import argparse
    return argparse.Namespace(out=str(out), arch=ARCH, epochs=epochs, seed=0, bs=32, lr=0.05, wd=5e-4, warmup=0,
                              sched="cosine", n_keys=2, per_key=5, eps=16.0, steps=2, restarts=1)


def test_driver_end_to_end_and_resume(tmp_path, monkeypatch):
    import json

    import keysel.run_phi as rp
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    a = _driver_args(tmp_path / "o")
    rp.main(a)
    o = tmp_path / "o"
    s = json.loads((o / "summary.json").read_text())
    assert s["bit_identical"] is True
    for n in ("R0", "R0b", "R1", "R1b_e1", "R1b_mid", "R2", "R3"):
        assert (o / n / "final.pt").exists(), n
    for n in ("R2", "R3"):
        assert (o / f"metrics_{n}" / "metrics.csv").exists() and "cos_eta_S0" in s[n]
    assert set(s["drift_curves"]) == {"R1", "R1b_e1", "R1b_mid", "R2", "R3"}
    assert all(len(v) == a.epochs + 1 for v in s["drift_curves"].values())
    # a second invocation reuses everything and yields identical numbers
    rp.main(a)
    assert json.loads((o / "summary.json").read_text())["R2"]["cos_eta_S0"] == s["R2"]["cos_eta_S0"]
    P = torch.load(o / "poisons.pt", weights_only=False)
    assert P["idx"].shape == (10,) and P["X_signed"].shape == (10, 3, 32, 32)
    assert (P["X_signed"] - fake_balanced()[0][P["idx"]].float() / 255).abs().max() <= 16 / 255 + 1e-6
    assert len(set(P["idx"].tolist())) == 10 and P["key_of"].tolist() == [0] * 5 + [1] * 5


def test_driver_injected_perturbation_matches_kick_size(tmp_path, monkeypatch):
    import keysel.run_phi as rp
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    a = _driver_args(tmp_path / "o")
    rp.main(a)
    from keysel.phi_metrics import drift_curve
    c = drift_curve(tmp_path / "o" / "R1b_e1", tmp_path / "o" / "R0")
    import json
    kick = json.loads((tmp_path / "o" / "summary.json").read_text())["mean_kick_norm"]
    assert c[0] == 0.0 and c[1] > 0.1 * kick  # injected at epoch 1: zero before, ~kick-sized right after


def test_figures_are_produced(tmp_path, monkeypatch):
    import keysel.run_phi as rp
    from keysel.phi_figures import make_figures
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    rp.main(_driver_args(tmp_path / "o"))
    out = make_figures(tmp_path / "o")
    assert (out / "phi_measurement.png").stat().st_size > 5000 and (out / "phi_lastk.png").exists()
