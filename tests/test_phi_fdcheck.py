"""CPU test of keysel.phi_fdcheck: on a smooth net the adjoint gradient matches real retraining finite differences."""
import torch

from keysel.phi_fdcheck import fd_check, make_directions
from tests.test_phi import D
from tests.test_phi_dyn import setup, signed_small


def test_fd_ratio_is_one_on_smooth_net(tmp_path):
    rp, kx, ky, final, q, X = setup(tmp_path)
    eps = 8 / 255
    Xs = signed_small(X, 0.01)
    rows = fd_check(rp, 1, Xs, kx, ky, eps, hs=[1e-3, 1e-4], chunk=3, log=lambda *_: None, offload=False)
    assert len(rows) == 4
    for r in rows:
        assert abs(r["analytic"]) > 1e-8
        assert abs(r["ratio"] - 1) < 2e-3 if r["h"] == 1e-4 else abs(r["ratio"] - 1) < 5e-2, r


def test_directions_are_feasible_in_both_signs():
    torch.manual_seed(0)
    X = torch.rand(4, 3, 8, 8, dtype=D)
    eps, h = 0.03, 0.01
    lo = torch.max(torch.full_like(X, -eps), -X)
    hi = torch.min(torch.full_like(X, eps), 1 - X)
    delta = torch.min(torch.max(0.05 * torch.randn_like(X), lo), hi)
    G = torch.randn_like(X)
    dirs, frac = make_directions(G, delta, X, eps, h)
    for v in dirs.values():
        assert float(v.abs().max()) <= 1 + 1e-12
        for s in (+1, -1):
            d2 = delta + s * h * v
            assert bool((d2 >= lo - 1e-12).all()) and bool((d2 <= hi + 1e-12).all())
    assert 0 < frac < 1
