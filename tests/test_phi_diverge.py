"""CPU tests of keysel.phi_diverge (perturbation growth screen), float64."""
import torch

from keysel.phi_diverge import perturbed_profile
from tests.test_phi import D
from tests.test_phi_transport import make


def test_zero_perturbation_stays_on_alice(tmp_path):
    rp = make(tmp_path)
    prof = perturbed_profile(rp, 1, torch.zeros(len(rp.pre(1)), dtype=D), window=0)
    assert max(prof.values()) < 1e-10 and "end" in prof


def test_small_perturbation_growth_matches_tangent(tmp_path):
    """Smooth net, tiny delta: kappa(end) equals ||Phi_k v|| for unit v (finite-difference regime)."""
    rp = make(tmp_path)
    k = 0
    g = torch.Generator().manual_seed(0)
    v = torch.randn(len(rp.pre(k)), generator=g, dtype=D)
    v = v / v.norm()
    eps = 1e-7
    prof = perturbed_profile(rp, k, eps * v, window=0)
    _, w, _ = rp.forward(k, v)
    assert abs(prof["end"] / eps - float(w.norm())) < 1e-4 * float(w.norm())


def test_window_limits_epochs(tmp_path):
    rp = make(tmp_path)
    prof = perturbed_profile(rp, 0, 1e-3 * torch.ones(len(rp.pre(0)), dtype=D), window=2)
    assert set(prof) == {1, 2}
