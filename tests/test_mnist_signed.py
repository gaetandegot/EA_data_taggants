import torch
from src.datasets import MNIST


def test_signed_mnist_override_keeps_float_precision():
    ds = MNIST.__new__(MNIST)
    ds.data = torch.zeros((2, 28, 28), dtype=torch.uint8)
    ds.targets = torch.tensor([3, 8])
    ds.transform = lambda x: (x - 0.13066373765468597) / 0.30810782313346863
    ds.target_transform = None
    signed = torch.full((1, 28, 28), 0.027)
    ds.signed_overrides = {1: signed}
    x, y = ds[1]
    assert y == 8
    assert torch.allclose(x, ds.transform(signed))
    clean, _ = ds[0]
    assert torch.allclose(clean, ds.transform(torch.zeros((1, 28, 28))))
