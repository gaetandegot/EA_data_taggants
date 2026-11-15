"""CPU tests of keysel.phi_dyn (problem B, adjoint over the window, Bob replay, eta_lin), float64, smooth net."""
import torch

from keysel.phi import batch_grad, set_flat
from keysel.phi_dyn import (adjoint_window, bob_window, craft_B, grad_delta, linear_eta, pdot_g, poison_lr,
                            split_like)
from keysel.phi_metrics import load_final
from tests.test_phi import D, DATA, PIDX
from tests.test_phi_transport import EPOCHS, make


def setup(tmp):
    rp = make(tmp)
    g = torch.Generator().manual_seed(3)
    kx = torch.rand(2, 3, 32, 32, generator=g, dtype=D)
    ky = torch.tensor([5, 7])
    final = load_final(tmp / "R0").to(D)
    q = batch_grad(rp.model, final, kx, ky)
    X = rp.Xf[PIDX].clone()
    return rp, kx, ky, final, q, X


def signed_small(X, scale):
    g = torch.Generator().manual_seed(1)
    return (X + scale * torch.sign(torch.randn(X.shape, generator=g, dtype=D))).clamp(0, 1)


def test_bob_with_clean_images_is_alice(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    thB, anch = bob_window(rp, 1, X)
    assert float((thB - final).norm()) < 1e-10
    assert all(a["drift"] < 1e-10 for a in anch)


def test_adjoint_formula_equals_forward_tangent(tmp_path):
    """q^T eta_lin from the forward tangent pass == - sum_e gamma_e P_e^T kick_e from one backward sweep."""
    rp, kx, ky, final, q, X = setup(tmp_path)
    k0 = 1
    Xs = signed_small(X, 0.05)
    el, kicks = linear_eta(rp, k0, Xs)
    P = adjoint_window(rp, k0, q)
    assert set(P) == set(range(k0, EPOCHS))
    adj = -sum(poison_lr(rp, e) * float(P[e] @ kicks[e]) for e in kicks)
    assert abs(float(q @ el) - adj) < 1e-9 * max(1.0, abs(adj))


def test_eta_lin_matches_true_eta_for_tiny_poisons(tmp_path):
    """Linear regime (tiny kicks, smooth net): eta_lin ~ eta."""
    rp, kx, ky, final, q, X = setup(tmp_path)
    Xs = signed_small(X, 1e-4)
    thB, _ = bob_window(rp, 1, Xs)
    eta = thB - final
    el, _ = linear_eta(rp, 1, Xs)
    assert float((el - eta).norm() / eta.norm()) < 1e-3


def test_pixel_gradient_matches_finite_differences(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    P = adjoint_window(rp, 1, q)
    e = 2
    theta, parts = rp.pre(e), split_like(rp.model, P[e])
    x, y = X[:3], rp.ytr[PIDX[:3]]
    d = 0.02 * torch.randn(x.shape, dtype=D)
    v = torch.randn(x.shape, dtype=D)
    g = grad_delta(rp.model, theta, parts, x, y, d)
    h = 1e-6
    f = lambda dd: -float(pdot_g(rp.model, theta, parts, x + dd, y, False))  # noqa: E731
    fd = (f(d + h * v) - f(d - h * v)) / (2 * h)
    assert abs(float((g * v).sum()) - fd) < 1e-5 * max(1.0, abs(fd))


def test_craft_B_improves_objective_and_respects_budget(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    k0 = 1
    P = adjoint_window(rp, k0, q)
    y = rp.ytr[PIDX]
    groups = [torch.arange(0, 3), torch.arange(3, 6)]
    eps = 8 / 255
    r = craft_B(rp, P, k0, X, y, groups, eps=eps, steps=40, restarts=2, gen=torch.Generator().manual_seed(0),
                chunk=3, log=lambda *_: None)
    assert all(fb < fz for fb, fz in zip(r["F_best"], r["F_zero"]))
    assert float(r["delta"].abs().max()) <= eps + 1e-12
    Xs = X + r["delta"]
    assert float(Xs.min()) >= 0 and float(Xs.max()) <= 1
