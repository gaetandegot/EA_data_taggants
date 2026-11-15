"""CPU tests of keysel.phi_transport (replay, tangent vs finite differences, adjoint identity, power iteration)."""
import torch

from keysel.phi import train_traj
from keysel.phi_metrics import load_final
from keysel.phi_transport import Replay, power_iteration
from tests.test_phi import DATA, PIDX, D

EPOCHS, BS = 4, 32


def make(tmp, arch="tinysmooth_gn"):
    train_traj(arch, 0, tmp / "R0", DATA, PIDX, X_signed=None, epochs=EPOCHS, bs=BS, lr=0.05, wd=5e-4,
               warmup_epochs=0, sched="cosine", device="cpu", dtype=D, log=lambda *_: None)
    return Replay(tmp, arch, 0, DATA, PIDX, EPOCHS, BS, 0.05, 5e-4, 0, "cosine", device="cpu", dtype=D)


def test_replay_reaches_alice_final_and_checkpoints(tmp_path):
    rp = make(tmp_path)
    th_end, _, _ = rp.forward(1, None)
    assert float((th_end - load_final(tmp_path / "R0").to(D)).norm()) < 1e-10
    assert max(d for _, d in rp.dev_log) < 1e-10


def test_tangent_matches_finite_differences(tmp_path):
    """Phi_k v = d/d eps of the replayed end state started at theta_start + eps v (smooth net)."""
    rp = make(tmp_path)
    k = 1
    g = torch.Generator().manual_seed(0)
    v = torch.randn(len(rp.pre(k)), generator=g, dtype=D)
    _, w, _ = rp.forward(k, v)
    eps = 1e-6
    ends = []
    for s in (+1, -1):
        theta = rp.start(k) + s * eps * v
        for e in range(k + 1, EPOCHS):
            plan = [rp.batch(e, j) for j in range(rp.n_steps)]
            for j, (x, y) in enumerate(plan):
                theta = rp.sgd(theta, x, y, rp.lr_step(e, j))
        ends.append(theta)
    fd = (ends[0] - ends[1]) / (2 * eps)
    assert float((w - fd).norm() / fd.norm()) < 1e-5


def test_adjoint_is_transpose_of_forward(tmp_path):
    rp = make(tmp_path)
    k = 0
    g = torch.Generator().manual_seed(1)
    n = len(rp.pre(k))
    v, u = torch.randn(n, generator=g, dtype=D), torch.randn(n, generator=g, dtype=D)
    _, Pv, _ = rp.forward(k, v)
    PTu = rp.adjoint(k, u)
    assert abs(float(u @ Pv) - float(PTu @ v)) < 1e-9 * max(1.0, abs(float(u @ Pv)))


def test_power_iteration_is_monotone_and_beats_random_start(tmp_path):
    rp = make(tmp_path)
    k = 0
    g = torch.Generator().manual_seed(2)
    v0 = torch.randn(len(rp.pre(k)), generator=g, dtype=D)
    res = power_iteration(rp, k, v0, iters=4, log=lambda *_: None)
    s = res["sigmas"]
    assert all(b >= a - 1e-9 for a, b in zip(s, s[1:]))
    assert s[-1] >= s[0]
    assert set(res["profile_first_vector"]) == set(range(k + 1, EPOCHS))
