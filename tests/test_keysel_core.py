import timm
import torch
import torch.nn.functional as F
from scipy import stats

from keysel.craft import apply_aug, craft_key, match_and_grad, sample_aug
from keysel.keys import make_key, make_keys
from keysel.models import Wrapped, build_cifar_model, disable_fused_attn, head_params
from keysel.scores import head_mask, key_grad, model_params, split_cos, sum_grad, zero_score

torch.set_num_threads(2)


def tiny_cnn(nc=5):
    torch.manual_seed(0)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
                                            torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
            self.fc = torch.nn.Linear(4, nc)

        def forward(self, x):
            return self.fc(self.body(x))
    return Wrapped(Tiny(), (0.5,) * 3, (0.25,) * 3)


def tiny_vit():
    from timm.models.vision_transformer import VisionTransformer
    torch.manual_seed(0)
    net = VisionTransformer(img_size=16, patch_size=8, embed_dim=16, depth=1, num_heads=2, num_classes=5)
    return Wrapped(disable_fused_attn(net), (0.5,) * 3, (0.5,) * 3)


# ---- keys
def test_keys_deterministic_and_independent_of_count():
    a = make_key(7, 0, 64, 224, 1000)
    b = make_key(7, 0, 64, 224, 1000)
    assert torch.equal(a[0], b[0]) and a[1] == b[1]
    xs, ys = make_keys([3, 7], 0, 64, 224, 1000)
    assert torch.equal(xs[1], a[0]) and ys[1] == a[1]
    assert not torch.equal(make_key(8, 0, 64, 224, 1000)[0], a[0])


def test_keys_blocky_range_labels():
    x, y = make_key(1, 0, 64, 224, 1000)
    assert x.shape == (3, 224, 224) and 0 <= x.min() and x.max() <= 1 and 0 <= y < 1000
    # nearest upsampling 64 -> 224: pixel (0,0) and (0,1) share a block (224/64 = 3.5 > 1)
    assert torch.equal(x[:, 0, 0], x[:, 1, 1])
    xs, ys = make_keys(range(3000), 0, 32, 32, 10)
    counts = torch.bincount(ys, minlength=10).float()
    assert stats.chisquare(counts.numpy()).pvalue > 1e-3


# ---- models
def test_cifar_models_shapes_and_params():
    exp = {"resnet18": (11.0e6, 11.4e6), "resnet20": (0.26e6, 0.30e6), "resnet56": (0.83e6, 0.90e6), "vgg11": (9.0e6, 9.4e6)}
    for name, (lo, hi) in exp.items():
        m = build_cifar_model(name, 100)
        n = sum(p.numel() for p in m.parameters())
        assert lo < n < hi, (name, n)
        assert m(torch.rand(2, 3, 32, 32)).shape == (2, 100)
        hp = head_params(m)
        assert len(hp) == 2 and hp[0].shape == (100, hp[0].shape[1])


def test_head_params_timm():
    assert len(head_params(tiny_vit())) == 2


# ---- cosines
def test_split_cos_matches_manual():
    m = tiny_cnn()
    p = model_params(m)
    x = torch.rand(3, 3, 8, 8)
    ga = key_grad(m, x[0], 1)
    gb = sum_grad(m, x, 1)
    mask = head_mask(m, p)
    full, head = split_cos(ga, gb, mask)
    fa = torch.cat([g.flatten() for g in ga]); fb = torch.cat([g.flatten() for g in gb])
    assert torch.allclose(full, F.cosine_similarity(fa, fb, dim=0), atol=1e-6)
    hid = {id(q) for q in head_params(m)}
    ha = torch.cat([g.flatten() for g, q in zip(ga, p) if id(q) in hid])
    hb = torch.cat([g.flatten() for g, q in zip(gb, p) if id(q) in hid])
    assert torch.allclose(head, F.cosine_similarity(ha, hb, dim=0), atol=1e-6)


def test_sum_grad_chunk_invariant():
    m = tiny_cnn()
    x = torch.rand(7, 3, 8, 8)
    a = sum_grad(m, x, 2, chunk=7)
    b = sum_grad(m, x, 2, chunk=2)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-6)
    r = zero_score(m, x[0], 2, x, chunk=3)
    assert -1 <= r["full"] <= 1 and 1 <= r["rank"] <= 5


# ---- aug
def test_aug_identity_and_flip():
    x = torch.rand(2, 3, 8, 8)
    assert apply_aug(x, None) is x
    aug = (torch.full((2,), 4), torch.full((2,), 4), torch.tensor([False, True]))
    out = apply_aug(x, aug)
    assert torch.allclose(out[0], x[0]) and torch.allclose(out[1], x[1].flip(-1))


def test_aug_seeded():
    a = sample_aug(5, 8, torch.Generator().manual_seed(1))
    b = sample_aug(5, 8, torch.Generator().manual_seed(1))
    c = sample_aug(5, 8, torch.Generator().manual_seed(2))
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not all(torch.equal(u, v) for u, v in zip(a, c))
    x = torch.rand(5, 3, 8, 8, requires_grad=True)
    apply_aug(x, a).sum().backward()
    assert x.grad is not None


# ---- exactness of the memory trick
def _check_modes(model, shape):
    torch.manual_seed(1)
    params = model_params(model); mask = head_mask(model, params)
    X = torch.rand(5, *shape)
    key = torch.rand(*shape)
    gk = key_grad(model, key, 3, params)
    delta = (torch.rand_like(X) - 0.5) * 0.1
    aug = sample_aug(5, shape[-1], torch.Generator().manual_seed(0))
    for a in (None, aug):
        f1, h1, g1 = match_and_grad(model, params, mask, gk, X, delta, 3, a, chunk=5, mode="direct")
        f2, h2, g2 = match_and_grad(model, params, mask, gk, X, delta, 3, a, chunk=2, mode="chunked")
        assert abs(f1 - f2) < 1e-5 and abs(h1 - h2) < 1e-5
        rel = (g1 - g2).norm() / (g1.norm() + 1e-12)
        assert rel < 1e-4, float(rel)


def test_chunked_equals_direct_cnn():
    _check_modes(tiny_cnn().eval(), (3, 8, 8))


def test_chunked_equals_direct_vit():
    _check_modes(tiny_vit().eval(), (3, 16, 16))


# ---- craft
def test_craft_constraints_and_progress():
    m = tiny_cnn().eval()
    X = torch.rand(6, 3, 8, 8)
    key = torch.rand(3, 8, 8)
    from keysel.craft import score_of
    params = model_params(m); mask = head_mask(m, params)
    gk = key_grad(m, key, 2, params)
    before = score_of(m, params, mask, gk, X, 2)[0]
    out = craft_key(m, key, 2, X, eps=16 / 255, steps=30, restarts=2, tau=0.1, augment=False)
    d = out["delta"]
    assert d.abs().max() <= 16 / 255 + 1e-6
    assert (X + d).min() >= -1e-6 and (X + d).max() <= 1 + 1e-6
    assert out["full"] > before
    assert out["full"] == max(out["all_full"]) and len(out["all_full"]) == 2


def test_craft_restarts_differ():
    m = tiny_cnn().eval()
    X = torch.rand(4, 3, 8, 8); key = torch.rand(3, 8, 8)
    out = craft_key(m, key, 1, X, steps=3, restarts=3, augment=True)
    assert len(set(round(v, 6) for v in out["all_full"])) > 1
