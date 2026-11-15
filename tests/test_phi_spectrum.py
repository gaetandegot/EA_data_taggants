"""CPU tests of keysel.phi_spectrum (Lanczos extremes, growth comparison), float64."""
import math

import torch

from keysel.models import build_cifar_model
from keysel.phi import flat, lr_at
from keysel.phi_lin import hvp
from keysel.phi_spectrum import compare, lanczos, log_factor, spearman
from tests.test_phi import DATA, D


def test_lanczos_extremes_on_known_spectrum():
    """Symmetric matrix with known eigenvalues (one negative outlier, one positive outlier, near-zero bulk)."""
    g = torch.Generator().manual_seed(0)
    n = 300
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=D))
    lam = torch.cat([torch.tensor([-2.0, 50.0], dtype=D), 1e-3 * torch.randn(n - 2, generator=g, dtype=D)])
    A = Q @ torch.diag(lam) @ Q.T
    r, r_half = lanczos(lambda v: A @ v, n, 40, dtype=D)
    assert abs(float(r[0]) + 2.0) < 1e-8 and abs(float(r[-1]) - 50.0) < 1e-8
    # Ritz values interlace: always inside the true spectrum
    assert float(r_half[0]) >= -2.0 - 1e-10 and float(r_half[-1]) <= 50.0 + 1e-10


def test_lanczos_full_dimension_recovers_all_eigenvalues():
    g = torch.Generator().manual_seed(1)
    n = 30
    B = torch.randn(n, n, generator=g, dtype=D)
    A = B + B.T
    r, _ = lanczos(lambda v: A @ v, n, n, dtype=D)
    assert torch.allclose(r, torch.linalg.eigvalsh(A), atol=1e-8)


def test_lanczos_with_hvp_matches_dense_hessian():
    """Extreme eigenvalues of a tiny GroupNorm net's batch Hessian: Lanczos (HVP) vs dense Hessian."""
    torch.manual_seed(0)
    m = build_cifar_model("tinysmooth_gn").to(D)
    th = flat(m).clone()
    x, y = DATA[0][:4].to(D) / 255, DATA[1][:4]
    P = len(th)
    H = torch.stack([hvp(m, th, x, y, torch.nn.functional.one_hot(torch.tensor(i), P).to(D)) for i in range(P)])
    assert torch.allclose(H, H.T, atol=1e-8)
    ev = torch.linalg.eigvalsh((H + H.T) / 2)
    r, _ = lanczos(lambda v: hvp(m, th, x, y, v), P, 120, dtype=D)
    assert abs(float(r[-1]) - float(ev[-1])) < 1e-6 * max(1.0, abs(float(ev[-1])))
    assert float(r[0]) >= float(ev[0]) - 1e-8          # interlacing: never below the true minimum
    assert float(r[0]) < 0.5 * float(ev[0]) if float(ev[0]) < 0 else True  # finds most of the negative edge


def test_log_factor():
    assert log_factor(0.1, 1.0, 10.0) == math.log(0.9)          # contracting: |1-0.1|=0.9 dominates |1-1|=0
    assert log_factor(0.1, -1.0, 10.0) == math.log(1.1)         # negative curvature expands
    assert log_factor(0.1, 0.5, 30.0) == math.log(2.0)          # lambda > 2/lr: overshoot


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [1, 3, 2, 4]) - 0.8) < 1e-12


def test_compare_recovers_exact_growth():
    """If |e_lin| grows exactly by the predicted factors, observed = predicted and both correlations are 1."""
    epochs, n_steps, lr, wd = 6, 5, 0.1, 5e-4
    total, warm = epochs * n_steps, n_steps
    spec, lin = [], [{"epoch": 0, "e_lin_norm": 0.0}, {"epoch": 1, "e_lin_norm": 1.0}]
    norm = 1.0
    for e in range(epochs):
        lmin, lmax = -0.5 * (epochs - e), 3.0 + e
        spec.append(dict(epoch=e, p_lmin=lmin, p_lmax=lmax, c_lmin=lmin, c_lmax=lmax))
        if e >= 1:
            g = log_factor(lr_at(e * n_steps + n_steps - 1, total, lr, warm, "cosine"), lmin + wd, lmax + wd)
            if e + 1 < epochs:
                g += sum(log_factor(lr_at((e + 1) * n_steps + j, total, lr, warm, "cosine"), lmin + wd, lmax + wd)
                         for j in range(n_steps - 1))
            norm *= math.exp(g)
            lin.append({"epoch": e + 1, "e_lin_norm": norm})
    rows, s = compare(spec, lin, n_steps, total, lr, warm, "cosine", wd, epochs)
    assert len(rows) == epochs - 1
    for r in rows:
        assert abs(r["ghat"] - r["g_obs"]) < 1e-10
    assert abs(s["spearman_ghat_gobs"] - 1) < 1e-12 and s["frac_obs_below_pred"] == 1.0
    assert abs(rows[0]["G_obs_remaining"] - math.log(norm)) < 1e-10
