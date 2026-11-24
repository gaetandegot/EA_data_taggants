"""Backdoor setting (Sleeper Agent, Souri et al. 2022): a random 8x8 patch pasted at a random position turns images of
a source class into the target class. The poisons are clean-label target-class training images; the attacker's
objective is the cross-entropy of the target class on TRIGGERED source images. Crafting uses the triggered source
TRAINING images; success is measured on the triggered source TEST images, never seen during crafting.
"""
import torch

PATCH = 8


def backdoor_spec(seed, nc=100, patch=PATCH, channels=3):
    """(source class, target class != source, patch (C,patch,patch) uniform in [0,1]), deterministic in seed."""
    g = torch.Generator().manual_seed(seed * 100003 + 8888)
    s = int(torch.randint(nc, (1,), generator=g))
    t = int((s + torch.randint(1, nc, (1,), generator=g)) % nc)
    return s, t, torch.rand(channels, patch, patch, generator=g)


def apply_trigger(X, patch, seed):
    """Paste `patch` at a uniformly random position (one per image, deterministic in seed) into float images X."""
    X = X.clone()
    n, _, H, W = X.shape
    p = patch.shape[-1]
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(H - p + 1, (n,), generator=g)
    cols = torch.randint(W - p + 1, (n,), generator=g)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        X[i, :, r:r + p, c:c + p] = patch.to(X.dtype)
    return X


def _float(X):
    return X.float() / 255 if X.dtype == torch.uint8 else X.float()


def triggered_sources(data, seed, split="train", nc=100):
    """Triggered images of the source class from the train or test split, with the target label."""
    s, t, patch = backdoor_spec(seed, nc)
    X, y = (data[0], data[1]) if split == "train" else (data[2], data[3])
    Xs = _float(X.cpu()[y.cpu() == s])
    stream = seed * 100003 + (1 if split == "train" else 2)  # different trigger positions for crafting and testing
    return apply_trigger(Xs, patch, stream), torch.full((len(Xs),), t, dtype=torch.long)
