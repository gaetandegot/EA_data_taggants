import torch

from keysel.data import augment_batch, signing_subset, to_float
from keysel.train_cifar import train


def fake_data(n=256, nc=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)
    y = torch.randint(0, nc, (n,), generator=g)
    return X, y, X[:64].clone(), y[:64].clone()


def test_augment_batch_shape_dtype():
    x = torch.rand(4, 3, 32, 32)
    out = augment_batch(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.dtype == x.dtype


def test_signing_subset_class_and_determinism():
    y = torch.arange(1000) % 10
    a = signing_subset(y, 3, 20, 0); b = signing_subset(y, 3, 20, 0)
    assert torch.equal(a, b) and (y[a] == 3).all() and len(set(a.tolist())) == 20


def test_train_resume_reproduces(tmp_path):
    data = fake_data()
    full = train("resnet20", 0, tmp_path / "a", epochs=2, bs=64, data=data, device="cpu", log=lambda *_: None)
    # simulate a kill after epoch 1: run 1 epoch worth, keep checkpoint, then resume
    import os
    import time
    os.environ["GPU_DEADLINE"] = str(time.time() + 0.0)  # deadline already passed after epoch 1
    try:
        train("resnet20", 0, tmp_path / "b", epochs=2, bs=64, data=data, device="cpu", log=lambda *_: None)
    except SystemExit as e:
        assert e.code == 75
    finally:
        del os.environ["GPU_DEADLINE"]
    assert (tmp_path / "b" / "checkpoint.pth").exists() and not (tmp_path / "b" / "final.pth").exists()
    res = train("resnet20", 0, tmp_path / "b", epochs=2, bs=64, data=data, device="cpu", log=lambda *_: None)
    for (k, u), (_, v) in zip(full.state_dict().items(), res.state_dict().items()):
        assert torch.allclose(u.float(), v.float(), atol=1e-5), k
