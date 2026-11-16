"""CPU tests of keysel.phi_gapdiag (identity q^T e_lin = pred, linear response vs real Bob, kink test), float64."""
import torch

from keysel.phi import batch_grad
from keysel.phi_dyn import objective_groups, split_like
from keysel.phi_gapdiag import kink_test, linear_response
from keysel.phi_recenter import adjoint_bob, bob_traj
from tests.test_phi import D, PIDX
from tests.test_phi_dyn import setup, signed_small


def test_identity_and_linear_regime(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    k0 = 1
    Xo = signed_small(X, 0.01)
    g = torch.Generator().manual_seed(9)
    Xn = (Xo + 1e-5 * torch.randn(Xo.shape, generator=g, dtype=D)).clamp(0, 1)
    th, starts, pres = bob_traj(rp, k0, Xo)
    qe = batch_grad(rp.model, th, kx, ky)
    Pb = adjoint_bob(rp, k0, Xo, qe, starts)
    ew = list(range(k0, rp.epochs))
    parts = {e: split_like(rp.model, Pb[e]) for e in ew}
    groups = [torch.arange(len(X))]
    y = rp.ytr[PIDX]
    Fn = objective_groups(rp, pres, parts, ew, X, y, Xn - X, groups, len(X), 6)[0]
    Fo = objective_groups(rp, pres, parts, ew, X, y, Xo - X, groups, len(X), 6)[0]
    th_end, ev, prof = linear_response(rp, k0, Xo, Xn)
    assert float((th_end - th).norm()) < 1e-10
    assert abs(float(qe @ ev) - (Fn - Fo)) < 1e-8 * max(1e-12, abs(Fn - Fo)) + 1e-14
    thN, _, presN = bob_traj(rp, k0, Xn)
    d = thN - th
    assert float((d - ev).norm() / ev.norm()) < 1e-3          # smooth net, tiny step: linear response is exact
    for e in prof:
        if e > k0:
            de = presN[e] - pres[e]
            assert float((de - prof[e]).norm() / prof[e].norm()) < 1e-3


def test_kink_test_on_smooth_net(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    Xo = signed_small(X, 0.01)
    th, starts, pres = bob_traj(rp, 1, Xo)
    g = torch.Generator().manual_seed(3)
    d = 1e-4 * torch.randn(len(th), generator=g, dtype=D)
    for po in (False, True):
        rows = kink_test(rp, 2, pres[2], Xo, d, (1.0, 0.25), poison_only=po, log=lambda *_: None)
        assert all(abs(r["ratio_fd_tangent"] - 1) < 1e-4 and r["cos_fd_tangent"] > 1 - 1e-8 for r in rows)


def test_even_part_slope_two_on_smooth_net(tmp_path):
    """Smooth net: the even part of the SGD map is second order (slope 2)."""
    from keysel.phi_evenpart import even_part
    rp, kx, ky, final, q, X = setup(tmp_path)
    Xo = signed_small(X, 0.01)
    th, starts, pres = bob_traj(rp, 1, Xo)
    g = torch.Generator().manual_seed(4)
    v = 1e-2 * torch.randn(len(th), generator=g, dtype=D)
    res = even_part(rp, 2, pres[2], Xo, v, (1.0, 0.25, 0.0625), log=lambda *_: None)
    for mode in ("poison_step", "span"):
        assert all(abs(r["slope_even"] - 2) < 0.1 for r in res[mode][1:]), res[mode]
