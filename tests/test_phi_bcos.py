"""CPU tests of keysel.phi_bcos (cosine objective of problem B), float64, smooth net."""
import torch

from keysel.phi_bcos import cos_and_grad, craft_Bcos, epoch_weights, objective_cos
from keysel.phi_dyn import adjoint_window
from tests.test_phi import D, PIDX
from tests.test_phi_dyn import setup, signed_small


def test_cos_gradient_matches_finite_differences(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    P = adjoint_window(rp, 1, q)
    e = 2
    y = rp.ytr[PIDX]
    groups = [torch.arange(0, 3), torch.arange(3, 6)]
    delta = 0.01 * torch.randn(X.shape, generator=torch.Generator().manual_seed(0), dtype=D)
    vals, g = cos_and_grad(rp, rp.pre(e), P[e], X, y, delta, groups, chunk=2)
    assert all(-1 <= v <= 1 for v in vals)
    v = torch.randn(X.shape, generator=torch.Generator().manual_seed(1), dtype=D)
    h = 1e-6
    f = lambda dd: -sum(cos_and_grad(rp, rp.pre(e), P[e], X, y, dd, groups, 2, need_grad=False)[0])  # noqa: E731
    fd = (f(delta + h * v) - f(delta - h * v)) / (2 * h)
    assert abs(float((g * v).sum()) - fd) < 1e-5 * max(1e-3, abs(fd)), (float((g * v).sum()), fd)


def test_craft_improves_cosine_and_respects_box(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    k0, eps = 1, 8 / 255
    P = adjoint_window(rp, k0, q)
    y = rp.ytr[PIDX]
    groups = [torch.arange(0, 3), torch.arange(3, 6)]
    r = craft_Bcos(rp, P, k0, X, y, groups, eps=eps, steps=40, restarts=2, gen=torch.Generator().manual_seed(0),
                   chunk=3, log=lambda *_: None)
    assert all(b < z for b, z in zip(r["obj_best"], r["obj_zero"]))
    assert float(r["delta"].abs().max()) <= eps + 1e-12
    assert abs(sum(r["weights"]) - 1) < 1e-12
    # trust region around a starting point
    d0 = signed_small(X, 0.01) - X
    r2 = craft_Bcos(rp, P, k0, X, y, groups, eps=eps, steps=20, restarts=1, gen=torch.Generator().manual_seed(0),
                    chunk=3, log=lambda *_: None, delta0=d0, radius=1e-3)
    assert float((r2["delta"] - d0).abs().max()) <= 1e-3 + 1e-12
    assert all(b <= i + 1e-12 for b, i in zip(r2["obj_best"], r2["obj_init"]))


def test_weights(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    P = adjoint_window(rp, 1, q)
    ew = list(range(1, rp.epochs))
    w = epoch_weights(rp, P, ew, "gamma_norm")
    u = epoch_weights(rp, P, ew, "uniform")
    assert abs(float(w.sum()) - 1) < 1e-12 and torch.allclose(u, torch.full_like(u, 1 / len(ew)))
    assert (w > 0).all()


def test_run_bcos_end_to_end(tmp_path, monkeypatch):
    """Tiny CPU run folder -> E2 (adjoint, B, controls) -> B-cos with both scopes, files and fields."""
    import json

    import keysel.data as kd
    import keysel.run_phi as rp
    from keysel.phi_bcos import run_bcos
    from keysel.phi_dyn import run_e2
    from tests.test_phi import _driver_args, fake_balanced
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    monkeypatch.setattr(kd, "load_cifar100", lambda device="cpu": fake_balanced())
    o = tmp_path / "o"
    rp.main(_driver_args(o))
    quiet = lambda *_: None  # noqa: E731
    run_e2(o, 1, restarts=1, steps=2, device="cpu", log=quiet)
    for scope in ("group", "global"):
        res = run_bcos(o, 1, scope, restarts=1, steps=2, device="cpu", log=quiet)
        f = o / "e2" / "k1" / f"results_Bcos_{scope}.json"
        assert f.exists() and json.loads(f.read_text())["tag"] == f"Bcos_{scope}"
        assert (o / "e2" / "k1" / f"poisons_Bcos_{scope}.pt").exists()
        assert set(res["linear_objective_F"]) == {"Bcos", "GM_R2", "B"}
        assert len(res["anchoring"]) == 2 and res["eta_norm"] > 0
    # B-cos with zero optimisation steps stays at its random start: poisons within the box
    P = torch.load(o / "e2" / "k1" / "poisons_Bcos_group.pt", weights_only=False)
    assert float(P["X_signed"].min()) >= 0 and float(P["X_signed"].max()) <= 1
