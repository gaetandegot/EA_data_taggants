"""CPU tests of the linearised dynamics (keysel.phi_lin), float64."""
import torch

from keysel.models import build_cifar_model
from keysel.phi import batch_grad, flat, kick_at, set_flat, train_traj
from keysel.phi_lin import hvp, linearized_eta, run_lin
from keysel.phi_metrics import cos, load_final
from tests.test_phi import ARCH, BS, DATA, PIDX, D, fake_balanced, signed

KW = dict(bs=BS, lr=0.05, wd=5e-4, warmup_epochs=0, sched="cosine", device="cpu", dtype=D, log=lambda *_: None)


def traj(tmp, name, X_signed, epochs, **over):
    kw = {**KW, **over}
    return train_traj(ARCH, 0, tmp / name, DATA, PIDX, X_signed=X_signed, epochs=epochs, **kw)


def lin(tmp, X_signed, epochs, **over):
    kw = {**KW, **over}
    kw.pop("log")
    return linearized_eta(ARCH, 0, DATA, PIDX, X_signed, epochs=epochs, ref=(tmp / "A", tmp / "B"),
                          log=lambda *_: None, **kw)


def run_pair(tmp, xs, epochs, **over):
    traj(tmp, "A", None, epochs, **over)
    traj(tmp, "B", xs, epochs, **over)
    return lin(tmp, xs, epochs, **over)


def test_hvp_matches_finite_differences():
    """Smooth net + seeded direction: a ReLU net would make the finite difference cross activation kinks."""
    torch.manual_seed(0)
    m = build_cifar_model("tinysmooth_gn").to(D)
    th = flat(m).clone()
    x, y = DATA[0][:8].to(D) / 255, DATA[1][:8]
    v = torch.randn_like(th, dtype=D)
    h = 1e-6
    fd = (batch_grad(m, th + h * v, x, y) - batch_grad(m, th - h * v, x, y)) / (2 * h)
    assert torch.allclose(hvp(m, th, x, y, v), fd, rtol=1e-4, atol=1e-8)


def test_hvp_is_linear_and_symmetric():
    torch.manual_seed(1)
    m = build_cifar_model(ARCH).to(D)
    th = flat(m).clone()
    x, y = DATA[0][:8].to(D) / 255, DATA[1][:8]
    u, v = torch.randn_like(th), torch.randn_like(th)
    assert torch.allclose(hvp(m, th, x, y, 2 * u + 3 * v), 2 * hvp(m, th, x, y, u) + 3 * hvp(m, th, x, y, v), atol=1e-10)
    assert abs(float(u @ hvp(m, th, x, y, v)) - float(v @ hvp(m, th, x, y, u))) < 1e-10


def test_one_epoch_eta_lin_is_exact(tmp_path):
    xs = signed()
    res = run_pair(tmp_path, xs, 1)
    eta = load_final(tmp_path / "B") - load_final(tmp_path / "A")
    assert torch.allclose(res["eta_lin"], eta, atol=1e-13, rtol=0)


def test_replay_reproduces_alice_bitwise(tmp_path):
    res = run_pair(tmp_path, signed(), 3)
    assert torch.equal(res["theta_A"], load_final(tmp_path / "A"))
    assert all(r.get("replay_dev", 0.0) == 0.0 for r in res["rows"])


def test_zero_poison_gives_zero_eta_lin(tmp_path):
    clean = DATA[0][PIDX].to(D) / 255
    res = run_pair(tmp_path, clean, 3)
    assert res["eta_lin"].abs().max() == 0
    assert all(r["e_lin_norm"] == 0 for r in res["rows"])


def test_rows_structure(tmp_path):
    res = run_pair(tmp_path, signed(), 3)
    assert [r["epoch"] for r in res["rows"]] == [0, 1, 2, 3]
    assert res["rows"][0]["e_lin_norm"] == 0 and res["rows"][1]["e_lin_norm"] > 0
    assert {"cos_elin_eta", "ratio_elin_eta", "rel_err", "eta_norm"} <= set(res["rows"][2])


SMOOTH = "tinysmooth_gn"


def pair_smooth(tmp, xs, epochs, **over):
    kw = {**KW, **over}
    log = kw.pop("log")
    for name, x in (("A", None), ("B", xs)):
        train_traj(SMOOTH, 0, tmp / name, DATA, PIDX, X_signed=x, epochs=epochs, log=log, **kw)
    res = linearized_eta(SMOOTH, 0, DATA, PIDX, xs, epochs=epochs, ref=(tmp / "A", tmp / "B"), log=log, **kw)
    return res, load_final(tmp / "B") - load_final(tmp / "A")


def test_smooth_net_eta_lin_error_is_second_order(tmp_path):
    """Smooth activations: relative error of eta_lin must scale ~ like the drift, i.e. shrink with the perturbation."""
    errs = {}
    for scale in (1.0, 0.1, 0.01):
        sub = tmp_path / f"s{scale}"
        sub.mkdir()
        res, eta = pair_smooth(sub, signed(scale=scale), 4, lr=0.2, sched="const")
        errs[scale] = float((res["eta_lin"] - eta).norm() / eta.norm())
    assert errs[0.1] < 0.3 * errs[1.0] and errs[0.01] < 0.3 * errs[0.1] and errs[0.01] < 1e-2, errs


def test_smooth_net_eta_lin_beats_identity_transport(tmp_path):
    from keysel.phi_metrics import compute_metrics
    xs = signed(scale=0.05)
    res, eta = pair_smooth(tmp_path, xs, 5, lr=0.3, sched="const", wd=0.0)
    _, s = compute_metrics(SMOOTH, tmp_path / "A", tmp_path / "B", DATA, PIDX, xs, device="cpu", dtype=D)
    err_lin = float((res["eta_lin"] - eta).norm() / eta.norm())
    S0 = None  # Phi=I error expressed through the cosine/ratio: ||S0 - eta||/||eta||
    err_id = (s["ratio_S0"] ** 2 - 2 * s["ratio_S0"] * s["cos_eta_S0"] + 1) ** 0.5
    assert err_lin < 0.2 * err_id and err_lin < 0.05, (err_lin, err_id)


def test_smooth_net_per_epoch_rows_track_true_eta(tmp_path):
    res, _ = pair_smooth(tmp_path, signed(scale=0.05), 4, lr=0.2, sched="const")
    rel = [r["rel_err"] for r in res["rows"][1:]]
    assert len(rel) == 4 and max(rel) < 0.05
    assert all(r.get("replay_dev", 0.0) == 0.0 for r in res["rows"])


def test_relu_net_linearisation_has_a_kink_floor(tmp_path):
    """Documented limitation: with ReLU the error does NOT vanish with the perturbation (kink crossings)."""
    sub = tmp_path
    res = run_pair(sub, signed(scale=0.01), 4, lr=0.2, sched="const")
    eta = load_final(sub / "B") - load_final(sub / "A")
    assert float((res["eta_lin"] - eta).norm() / eta.norm()) > 1e-3  # far above the smooth-net error ~1e-4


def test_driver_run_lin_end_to_end(tmp_path, monkeypatch):
    import json

    import keysel.run_phi as rp
    import keysel.data as kd
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    monkeypatch.setattr(kd, "load_cifar100", lambda device="cpu": fake_balanced())
    from tests.test_phi import _driver_args
    a = _driver_args(tmp_path / "o")
    rp.main(a)
    s = run_lin(tmp_path / "o", ("R2", "R3"), device="cpu", log=lambda *_: None)
    assert set(s) == {"R2", "R3"} and (tmp_path / "o" / "lin_R2" / "rows.csv").exists()
    assert json.loads((tmp_path / "o" / "lin_R3" / "summary.json").read_text())["replay_bit_identical"] is True
    assert s["R2"]["cos_elin_eta"] > 0.5
