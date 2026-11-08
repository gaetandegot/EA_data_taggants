"""Models: CIFAR ResNets/VGG, timm loader, input-normalising wrapper, last-layer params."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wrapped(nn.Module):
    """Takes images in [0,1]; normalises internally so gradients wrt pixels are in pixel space."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def head_params(model):
    """Parameters of the final classification layer."""
    net = model.net if isinstance(model, Wrapped) else model
    if hasattr(net, "get_classifier"):
        head = net.get_classifier()
    elif hasattr(net, "fc"):
        head = net.fc
    else:  # e.g. the repo's MNISTCNN: last Linear layer
        head = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
    return list(head.parameters())


def make_norm(norm, c):
    """'bn' = BatchNorm2d; 'gn1' = GroupNorm(1, c) (LayerNorm over C,H,W: per-image, no running stats)."""
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm == "gn1":
        return nn.GroupNorm(1, c)
    raise ValueError(norm)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm="bn"):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = make_norm(norm, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = make_norm(norm, cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), make_norm(norm, cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.shortcut(x))


class CifarResNet18(nn.Module):
    def __init__(self, num_classes=100, norm="bn"):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = make_norm(norm, 64)
        layers, cin = [], 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
            layers.append(BasicBlock(cin, cout, stride, norm))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(F.relu(self.bn(self.conv(x))))
        # mean == adaptive_avg_pool2d(x, 1); its CUDA backward is non-deterministic
        return self.fc(x.mean((2, 3)))


class TinyCNNGN(nn.Module):
    """Small GroupNorm(1) CNN for tests and fast prototyping (~25k params)."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.n1 = nn.GroupNorm(1, 16)
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1, bias=False)
        self.n2 = nn.GroupNorm(1, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.n1(self.c1(x)))
        x = F.relu(self.n2(self.c2(x)))
        return self.fc(x.mean((2, 3)))


class _OptA(nn.Module):
    """Parameter-free shortcut of He et al. (subsample + zero-pad channels)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.pad = (cout - cin) // 2

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.shortcut = _OptA(cin, cout) if (stride != 1 or cin != cout) else nn.Identity()

    def forward(self, x):
        o = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(o + self.shortcut(x))


class CifarResNetN(nn.Module):
    """ResNet-(6n+2) for CIFAR (n=3 -> 20, n=9 -> 56)."""

    def __init__(self, n, num_classes=100):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(_Block(cin, cout, stride if i == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layers(F.relu(self.bn(self.conv(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class CifarVGG11(nn.Module):
    CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, num_classes=100):
        super().__init__()
        layers, cin = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_cifar_model(name, num_classes=100):
    nets = {
        "resnet18": lambda: CifarResNet18(num_classes),
        "resnet18_gn": lambda: CifarResNet18(num_classes, norm="gn1"),
        "tinycnn_gn": lambda: TinyCNNGN(num_classes),
        "resnet20": lambda: CifarResNetN(3, num_classes),
        "resnet56": lambda: CifarResNetN(9, num_classes),
        "vgg11": lambda: CifarVGG11(num_classes),
    }
    return Wrapped(nets[name](), CIFAR100_MEAN, CIFAR100_STD)


def build_timm_model(name, pretrained=True):
    """timm model wrapped with its own normalisation (input in [0,1])."""
    import timm
    from timm.data import resolve_data_config
    net = disable_fused_attn(timm.create_model(name, pretrained=pretrained))
    cfg = resolve_data_config({}, model=net)
    return Wrapped(net, cfg["mean"], cfg["std"]), cfg


def disable_fused_attn(model):
    """Fused SDPA has no double-backward; needed for gradient matching on ViTs."""
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    return model
