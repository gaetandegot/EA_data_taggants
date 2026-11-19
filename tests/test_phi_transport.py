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


def test_merged_poison_batch_larger_than_bs(tmp_path):
    """More poisons than one batch (here 40 > 32): ONE poison step with the lr of ceil(40/32) = 2 batches.
    Alice (train_traj) and the Replay apply the same scaled step (faithful replay, chunked gradients), and the
    merged step is first-order equal to two consecutive steps on the two halves."""
    from keysel.phi import lr_at, poison_scale
    pidx = torch.arange(50, 90)
    train_traj("tinysmooth_gn", 0, tmp_path / "R0", DATA, pidx, X_signed=None, epochs=3, bs=BS, lr=0.05, wd=5e-4,
               warmup_epochs=0, sched="cosine", device="cpu", dtype=D, log=lambda *_: None)
    rp = Replay(tmp_path, "tinysmooth_gn", 0, DATA, pidx, 3, BS, 0.05, 5e-4, 0, "cosine", device="cpu", dtype=D)
    assert rp.pscale == poison_scale(40, BS) == 2
    j = rp.n_steps - 1
    assert rp.lr_step(1, j) == 2 * lr_at(rp.n_steps + j, rp.total, 0.05, 0, "cosine")
    th_end, _, _ = rp.forward(0, None)
    assert float((th_end - load_final(tmp_path / "R0").to(D)).norm()) < 1e-10
    assert max(d for _, d in rp.dev_log) < 1e-10
    # first-order equivalence with two separate steps of gamma on the halves
    x, y = rp.batch(1, j)
    th0 = rp.pre(1)
    gam = 1e-3
    merged = rp.sgd(th0, x, y, 2 * gam)
    rp_bs = rp.bs
    rp.bs = 10 ** 9  # no chunking: plain steps on the halves
    two = rp.sgd(rp.sgd(th0, x[:20], y[:20], gam), x[20:], y[20:], gam)
    rp.bs = rp_bs
    step = float((merged - th0).norm())
    assert float((merged - two).norm()) < 1e-2 * step  # differs only at second order in gamma
