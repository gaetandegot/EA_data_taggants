"""CIFAR-100 held on the GPU, plus GPU crop/flip augmentation for training."""
from pathlib import Path

import numpy as np
import torch

from .craft import apply_aug, sample_aug

CIFAR_ROOT = "/Data/gaetan.degot/cifar100"


def load_cifar100(root=CIFAR_ROOT, device="cpu"):
    """Return (Xtr uint8 N,3,32,32, ytr, Xte, yte) on `device`.

    Reads the HF parquet files (train.parquet/test.parquet in `root`) and caches decoded npz.
    """
    out = []
    for split in ("train", "test"):
        cache = Path(root) / f"{split}.npz"
        if not cache.exists():
            import io

            import pyarrow.parquet as pq
            from PIL import Image
            t = pq.read_table(Path(root) / f"{split}.parquet").to_pydict()
            X = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
            np.savez(cache, X=X, y=np.asarray(t["fine_label"]))
        z = np.load(cache)
        X = torch.from_numpy(z["X"]).permute(0, 3, 1, 2).contiguous()
        out += [X.to(device), torch.from_numpy(z["y"]).long().to(device)]
    return tuple(out)


def to_float(Xu8):
    return Xu8.float().div_(255)


def augment_batch(x, gen, pad=4):
    """Random crop (pad, reflect) + flip on a float batch, on x's device."""
    return apply_aug(x, sample_aug(len(x), x.shape[-1], gen, pad=pad, device=x.device), pad)


def signing_subset(ytr, cls, n, seed):
    """Uniformly random n training indices of class `cls` (deterministic in seed, cls)."""
    idx = (ytr.cpu() == cls).nonzero().flatten()
    g = torch.Generator().manual_seed(seed * 100003 + int(cls))
    return idx[torch.randperm(len(idx), generator=g)[:n]]


def poison_subsets(ytr, ky, n, seed, pick="any"):
    """Disjoint training indices for each key, n per key (list of tensors, one per key; deterministic in seed).

    pick='any':    uniformly random images of ANY class (no link between a poison's label and its key's target).
    pick='target': random images of the key's target class (the Data Taggants clean-label rule); keys that share a
                   target class get disjoint images (signing_subset order, already-used images skipped).
    """
    ytr = ytr.cpu()
    if pick == "any":
        g = torch.Generator().manual_seed(seed * 100003 + 77777)
        perm = torch.randperm(len(ytr), generator=g)
        return [perm[k * n:(k + 1) * n] for k in range(len(ky))]
    if pick != "target":
        raise ValueError(f"unknown pick {pick!r}")
    used, out = set(), []
    for c in ky.tolist():
        cand = signing_subset(ytr, int(c), len(ytr), seed)
        idx = torch.tensor([i for i in cand.tolist() if i not in used][:n])
        assert len(idx) == n, f"class {c} has fewer than {n} unused images"
        used.update(idx.tolist())
        out.append(idx)
    return out
