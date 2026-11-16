"""CPU test of keysel.phi_gm8 and the gm8 start / stop marker of phi_recenter on a tiny run folder."""
import json

import torch


def test_gm8_and_recenter_start(tmp_path, monkeypatch):
    import keysel.data as kd
    import keysel.run_phi as rp
    from keysel.phi_gm8 import run_gm8
    from keysel.phi_recenter import run_recenter
    from tests.test_phi import _driver_args, fake_balanced
    monkeypatch.setattr(rp, "load_cifar100", lambda device="cpu": fake_balanced())
    monkeypatch.setattr(kd, "load_cifar100", lambda device="cpu": fake_balanced())
    o = tmp_path / "o"
    rp.main(_driver_args(o))
    quiet = lambda *_: None  # noqa: E731
    P0 = torch.load(o / "poisons.pt", weights_only=False)
    P8 = run_gm8(o, restarts=2, device="cpu", log=quiet)
    assert torch.equal(P8["idx"], P0["idx"]) and P8["X_signed"].shape == P0["X_signed"].shape
    assert float(P8["X_signed"].min()) >= 0 and float(P8["X_signed"].max()) <= 1
    assert (o / "gm8" / "craft.json").exists()
    h = run_recenter(o, 1, iters=1, steps=2, restarts=1, device="cpu", log=quiet, ladder=(0.5,), init="gm8",
                     stop_gain=1e9)
    f = o / "e2" / "k1" / "recenter_gm8" / "history.json"
    assert f.exists() and json.loads(f.read_text()).get("stopped")
    n = len(json.loads(f.read_text())["attempts"])
    run_recenter(o, 1, iters=1, steps=2, restarts=1, device="cpu", log=quiet, ladder=(0.5,), init="gm8", stop_gain=1e9)
    assert len(json.loads(f.read_text())["attempts"]) == n      # stopped ladders are not re-run
