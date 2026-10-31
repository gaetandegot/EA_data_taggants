"""Seeded key generation: uniform noise images and i.i.d. uniform labels."""
import hashlib

import torch
import torch.nn.functional as F


def key_generator(seed, key_id):
    """One independent RNG stream per (seed, key_id)."""
    h = hashlib.sha256(f"{seed}/{key_id}".encode()).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
    return g


def make_key(key_id, seed, noise_size, out_res, num_classes, channels=3):
    """Return (x, y): x in [0,1], shape (C,out_res,out_res), blocky if noise_size < out_res."""
    g = key_generator(seed, key_id)
    x = torch.rand(1, channels, noise_size, noise_size, generator=g)
    if noise_size != out_res:
        x = F.interpolate(x, size=(out_res, out_res), mode="nearest")
    y = int(torch.randint(num_classes, (1,), generator=g))
    return x[0], y


def make_keys(key_ids, seed, noise_size, out_res, num_classes, channels=3):
    xs, ys = zip(*[make_key(i, seed, noise_size, out_res, num_classes, channels) for i in key_ids])
    return torch.stack(xs), torch.tensor(ys)
