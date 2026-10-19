from types import SimpleNamespace

import torch

from src.models import build_model
from src.transforms import build_transforms


def test_mnist_cnn_forward_backward():
    args = SimpleNamespace(dataset='MNIST', model='mnist_cnn', input_size=28,
                           nb_classes=10, device=torch.device('cpu'),
                           distributed=False)
    transform_train, transform_val = build_transforms(args)
    x = torch.rand(4, 1, 28, 28)
    x = transform_train(x)
    assert transform_val(torch.rand(1, 28, 28)).shape == (1, 28, 28)
    model = build_model(args)
    logits = model(x)
    assert logits.shape == (4, 10)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert model.features[0].weight.grad is not None
