"""SGD with momentum in the late-window machinery (CPU, float64, tiny smooth network):
- a momentum Replay reproduces Alice's momentum training exactly (state = [theta, v]);
- adjoint_step is the transpose of tangent_step (dot-product test);
- the exact first-order gradient w.r.t. the poisons (adjoint along Bob + the crafting gradient, as in the ladder)
  matches a central finite difference of real Bob trainings, with and without momentum."""
import pytest
import torch

from keysel.phi import _load, train_traj
from keysel.phi_dyn import grad_delta, poison_lr, split_like
from keysel.phi_recenter import adjoint_bob, bob_traj
from keysel.phi_transport import Replay
from keysel.phi import batch_grad
from tests.test_phi_seed import fake_data

ARCH, EPOCHS, BS, LR, WD = "tinysmooth_gn", 3, 512, 0.05, 5e-4
DT = torch.float64


def _setup(tmp_path, mom):
    data = fake_data(n_per=30)
    pidx = torch.arange(0, 3000, 200)[:12]  # 12 poison positions
    run = tmp_path / f"m{mom}"
    train_traj(ARCH, 3, run / "R0", data, pidx, epochs=EPOCHS, bs=BS, lr=LR, wd=WD, device="cpu", dtype=DT,
               log=lambda s: None, momentum=mom)
    rp = Replay(run, ARCH, 3, data, pidx, EPOCHS, BS, LR, WD, 1, "cosine", device="cpu", dtype=DT, momentum=mom)
    return rp, data, pidx, run


@pytest.mark.parametrize("mom", [0.0, 0.9])
def test_replay_reproduces_alice(tmp_path, mom):
    rp, _, _, run = _setup(tmp_path, mom)
    th_end, _, _ = rp.forward(0)
    assert all(d == 0.0 for _, d in rp.dev_log)  # at every poison step the replay sits exactly on Alice's pre_e
    final = _load(run / "R0" / "final.pt")["theta"].to(DT)
    assert torch.equal(rp.theta_of(th_end), final)
    if mom:
        assert len(rp.pre(1)) == 2 * rp.n_par and float(rp.pre(1)[rp.n_par:].norm()) > 0


def test_adjoint_is_transpose_of_tangent(tmp_path):
    rp, _, _, _ = _setup(tmp_path, 0.9)
    g = torch.Generator().manual_seed(0)
    state = rp.pre(1)
    x, y = rp.batch(1, 0)
    w = torch.randn(len(state), generator=g, dtype=DT)
    p = torch.randn(len(state), generator=g, dtype=DT)
    lhs = float(rp.adjoint_step(state, p, x, y, 0.03) @ w)
    rhs = float(p @ rp.tangent_step(state, w, x, y, 0.03))
    assert abs(lhs - rhs) <= 1e-9 * max(1.0, abs(lhs))


@pytest.mark.parametrize("mom", [0.0, 0.9])
def test_exact_gradient_matches_finite_difference(tmp_path, mom):
    rp, data, pidx, _ = _setup(tmp_path, mom)
    k0 = 1
    g = torch.Generator().manual_seed(1)
    kx = torch.rand(4, 3, 32, 32, generator=g, dtype=DT)
    ky = torch.tensor([1, 5, 7, 9])
    X = rp.Xf[pidx].clone()
    y = rp.ytr[pidx]
    X = (X + 0.02 * torch.randn(X.shape, generator=g, dtype=DT)).clamp(0.05, 0.95)  # interior point

    def J(Xs):
        th = rp.theta_of(bob_traj(rp, k0, Xs)[0])
        set_th = th
        from keysel.phi import set_flat
        set_flat(rp.model, set_th)
        rp.model.train()
        return float(torch.nn.functional.cross_entropy(rp.model(kx), ky))

    thB, starts, pres = bob_traj(rp, k0, X)
    q = rp.pad(batch_grad(rp.model, rp.theta_of(thB), kx, ky))
    P = adjoint_bob(rp, k0, X, q, starts)
    G = torch.zeros_like(X)
    for e in range(k0, rp.epochs):
        parts = split_like(rp.model, P[e])
        G += poison_lr(rp, e) / len(X) * grad_delta(rp.model, rp.theta_of(pres[e]), parts, X, y, torch.zeros_like(X))
    D = torch.randn(X.shape, generator=g, dtype=DT)
    D /= D.norm()
    h = 1e-3
    fd = (J(X + h * D) - J(X - h * D)) / (2 * h)
    an = float((G * D).sum())
    assert abs(an) > 1e-8
    assert abs(fd - an) <= 1e-4 * abs(an) + 1e-9, (fd, an)
