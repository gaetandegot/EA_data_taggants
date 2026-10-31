"""Quick timing of one training epoch and one crafting step on CIFAR-100."""
import time

import torch

from .craft import match_and_grad
from .data import load_cifar100, signing_subset, to_float
from .keys import make_key
from .models import build_cifar_model
from .scores import head_mask, key_grad, model_params, zero_score
from .train_cifar import train


def sync():
    torch.cuda.synchronize()


if __name__ == "__main__":
    dev = "cuda"
    data = load_cifar100(device=dev)
    for arch in ("resnet18", "resnet56", "vgg11"):
        t = time.time()
        train(arch, 0, f"/Data/gaetan.degot/keysel/timing/{arch}", epochs=2, data=data)
        sync(); print(arch, "2 epochs incl. eval+startup", round(time.time() - t, 1), "s", flush=True)
    m = build_cifar_model("resnet18").to(dev).eval()
    Xtr, ytr = data[0], data[1]
    kx, ky = make_key(0, 0, 32, 32, 100)
    kx = kx.to(dev)
    X = to_float(Xtr[signing_subset(ytr, ky, 50, 0).to(dev)])
    params = model_params(m); mask = head_mask(m, params)
    gk = key_grad(m, kx, ky, params)
    delta = torch.zeros_like(X)
    for mode in ("direct", "chunked"):
        match_and_grad(m, params, mask, gk, X, delta, ky, None, mode=mode)
        sync(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        for _ in range(10):
            match_and_grad(m, params, mask, gk, X, delta, ky, None, mode=mode)
        sync(); print(f"craft step {mode}: {(time.time() - t) / 10 * 1000:.0f} ms, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    t = time.time(); zero_score(m, kx, ky, X); sync(); print("zero score", round(time.time() - t, 3), "s")
