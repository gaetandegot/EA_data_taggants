"""CPU tests of keysel.phi_recenter (adjoint along Bob = exact gradient of the real objective), float64, smooth net."""
import torch
import torch.nn.functional as F

from keysel.phi import batch_grad
from keysel.phi_dyn import craft_B, grad_delta, poison_lr, split_like
from keysel.phi_recenter import adjoint_bob, bob_traj
from tests.test_phi import D, PIDX
from tests.test_phi_dyn import setup, signed_small


def test_bob_traj_with_clean_images_is_alice(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    th, starts, pres = bob_traj(rp, 1, X)
    assert float((th - final).norm()) < 1e-10
    assert float((pres[2] - rp.pre(2)).norm()) < 1e-10 and float((starts[2] - rp.start(1)).norm()) < 1e-10


def key_ce(rp, kx, ky, theta):
    from keysel.phi import set_flat
    set_flat(rp.model, theta)
    rp.model.eval()
    out = float(F.cross_entropy(rp.model(kx), ky))
    rp.model.train()
    return out


def test_adjoint_along_bob_gives_exact_pixel_gradient(tmp_path):
    """Directional derivative of the REAL J(theta^B_N(delta)) (central finite difference, full re-training)
    equals sum_e gamma_e/|B| * grad_delta(-P_e^T g) . v computed from the adjoint along Bob's path."""
    rp, kx, ky, final, q, X = setup(tmp_path)
    k0 = 1
    Xs = signed_small(X, 0.03)
    th, starts, pres = bob_traj(rp, k0, Xs)
    q_end = batch_grad(rp.model, th, kx, ky)
    Pb = adjoint_bob(rp, k0, Xs, q_end, starts)
    y = rp.ytr[PIDX]
    g = torch.Generator().manual_seed(5)
    v = torch.randn(Xs.shape, generator=g, dtype=D)
    analytic = 0.0
    for e in pres:
        gd = grad_delta(rp.model, pres[e], split_like(rp.model, Pb[e]), Xs, y, torch.zeros_like(Xs))
        analytic += poison_lr(rp, e) / len(Xs) * float((gd * v).sum())
    h = 1e-6
    jp = key_ce(rp, kx, ky, bob_traj(rp, k0, Xs + h * v)[0])
    jm = key_ce(rp, kx, ky, bob_traj(rp, k0, Xs - h * v)[0])
    fd = (jp - jm) / (2 * h)
    assert abs(analytic - fd) < 1e-5 * max(1e-3, abs(fd)), (analytic, fd)


def test_trust_region_and_budget(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    k0, eps, r = 1, 8 / 255, 2 / 255
    Xs = signed_small(X, 0.02)
    th, starts, pres = bob_traj(rp, k0, Xs)
    Pb = adjoint_bob(rp, k0, Xs, batch_grad(rp.model, th, kx, ky), starts)
    d0 = Xs - X
    res = craft_B(rp, Pb, k0, X, rp.ytr[PIDX], [torch.arange(0, 3), torch.arange(3, 6)], eps=eps, steps=30,
                  restarts=1, tau=r / 2, gen=torch.Generator().manual_seed(0), chunk=3, log=lambda *_: None,
                  thetas=pres, delta0=d0, radius=r)
    assert float((res["delta"] - d0).abs().max()) <= r + 1e-12
    assert float(res["delta"].abs().max()) <= eps + 1e-12
    assert all(fb <= fi + 1e-12 for fb, fi in zip(res["F_best"], res["F_init"]))


def test_adjoint_offload_gives_identical_result(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    Xs = signed_small(X, 0.03)
    th, starts, pres = bob_traj(rp, 1, Xs)
    q_end = batch_grad(rp.model, th, kx, ky)
    a = adjoint_bob(rp, 1, Xs, q_end, starts)
    b = adjoint_bob(rp, 1, Xs, q_end, starts, offload=True)
    assert all(torch.equal(a[e], b[e]) for e in a)


def test_next_ladder():
    from keysel.phi_recenter import next_ladder
    assert next_ladder(0.25) == [0.125, 0.25, 0.5]
    assert next_ladder(0.5) == [0.125, 0.25, 0.5]          # 2r is out of range: extend downwards
    assert next_ladder(1 / 32) == [1 / 32, 1 / 16, 1 / 8]  # r/2 is out of range: extend upwards
    assert all(1 / 32 - 1e-12 <= x <= 0.5 + 1e-12 for x in next_ladder(0.125))
