"""CPU tests of keysel.phi_switches (switch counting on a small ReLU net)."""
import torch

from keysel.models import build_cifar_model
from keysel.phi import flat
from keysel.phi_switches import count_switches, relu_signs
from tests.test_phi import D, DATA


def test_switches_zero_at_zero_and_grow_with_h():
    torch.manual_seed(0)
    m = build_cifar_model("resnet18_gn").to(D)
    th = flat(m).clone()
    x = DATA[0][:2].to(D) / 255
    g = torch.Generator().manual_seed(1)
    v = torch.randn(len(th), generator=g, dtype=D)
    v = v / v.norm()
    base = relu_signs(m, th, x)
    assert count_switches(base, relu_signs(m, th, x))[0] == 0
    n = [count_switches(base, relu_signs(m, th + h * v, x))[0] for h in (1e-1, 1e-2, 1e-3)]
    assert n[0] >= n[1] >= n[2] and n[0] > 0
    # the activation calls were restored: the model output is unchanged after counting
    m2 = build_cifar_model("resnet18_gn").to(D)
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m(x), m2(x))
