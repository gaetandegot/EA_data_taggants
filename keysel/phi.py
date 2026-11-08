"""Phi=I measurement: deterministic 'poisons last' SGD trajectories for Alice (clean) and Bob (signed).

Setting (see docs/phi_identity_measurement_plan.md): plain SGD (momentum 0), no augmentation, fp32 (or
fp64 for tests), per-image normalisation (GroupNorm(1)), and every epoch ends with ONE batch made only of
the signing images. Alice trains on the clean originals of those images at the same positions, Bob on the
signed versions, so the two trajectories differ only through the poison kicks.

Just before each poison batch the parameters are saved (`traj/pre_XXX.pt`, flat vector + learning rate).
"""
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .models import build_cifar_model


def _load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)  # local trusted files


def set_deterministic():
    """Bitwise-reproducible kernels, no TF32."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def deadline_hit(margin_s=0.0):
    d = os.environ.get("GPU_DEADLINE")
    return d is not None and time.time() + margin_s > float(d)


def lr_at(step, total, lr, warmup_steps=0, kind="cosine"):
    """Learning rate used at global step `step` (0-based): linear warmup, then constant or cosine to 0."""
    if warmup_steps and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if kind == "const":
        return lr
    if kind == "cosine":
        t = (step - warmup_steps) / max(1, total - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    raise ValueError(kind)


def epoch_plan(n_train, poison_idx, epoch, seed, bs):
    """Index batches of one epoch: shuffled clean images in batches of `bs`, then the poison batch last.

    Independent of the image values, so Alice and Bob share exactly the same plan.
    """
    pidx = torch.as_tensor(poison_idx, dtype=torch.long).cpu()
    mask = torch.ones(n_train, dtype=torch.bool)
    mask[pidx] = False
    clean = mask.nonzero().flatten()
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    clean = clean[torch.randperm(len(clean), generator=g)]
    batches = [clean[i:i + bs] for i in range(0, len(clean), bs)]
    if len(pidx):
        if len(pidx) > bs:
            raise ValueError(f"{len(pidx)} poisons do not fit one batch of {bs}")
        batches.append(pidx)
    return batches


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


@torch.no_grad()
def set_flat(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(vec[i:i + n].view_as(p))
        i += n


def batch_grad(model, theta, x, y):
    """Flat gradient of the mean cross-entropy of batch (x, y) at parameters `theta` (no weight decay)."""
    set_flat(model, theta.to(next(model.parameters()).device))
    model.train()
    params = list(model.parameters())
    loss = F.cross_entropy(model(x), y)
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params)])


def kick_at(model, theta, x_clean, x_signed, y):
    """Poison kick Delta g(theta) = grad L(signed batch) - grad L(clean batch) (flat vector)."""
    return batch_grad(model, theta, x_signed, y) - batch_grad(model, theta, x_clean, y)


def random_direction(n, norm, seed, device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g, dtype=torch.float64)
    return (v * (norm / v.norm())).to(device=device, dtype=dtype)


def train_traj(arch, seed, out_dir, data, poison_idx, X_signed=None, epochs=40, bs=512, lr=0.05, wd=5e-4,
               warmup_epochs=1, sched="cosine", init_noise=0.0, inject=None, device="cuda",
               dtype=torch.float32, save_pre=True, log=print, deterministic=True):
    """Train one trajectory; returns the final model. Resumable (per-epoch checkpoint).

    data = (Xtr uint8, ytr, Xte, yte); poison_idx = training positions placed in the last batch of every epoch;
    X_signed = float images (n_p,3,32,32) in [0,1] used for those positions (None = Alice, clean originals).
    init_noise = relative norm of a Gaussian perturbation of the initial parameters.
    inject = (epoch, vec): vec is added to the parameters at the start of that epoch.
    """
    if deterministic:
        set_deterministic()
    out = Path(out_dir)
    (out / "traj").mkdir(parents=True, exist_ok=True)
    final = out / "final.pt"
    Xtr, ytr, Xte, yte = data
    pidx = torch.as_tensor(poison_idx, dtype=torch.long).cpu()
    torch.manual_seed(seed)
    model = build_cifar_model(arch).to(device=device, dtype=dtype)
    if init_noise:
        theta0 = flat(model)
        set_flat(model, theta0 + random_direction(len(theta0), init_noise * float(theta0.norm()),
                                                  seed + 12345, device, dtype))
    if final.exists():
        set_flat(model, _load(final, map_location=device)["theta"].to(dtype))
        return model
    Xf = Xtr.to(device).to(dtype) / 255 if Xtr.dtype == torch.uint8 else Xtr.to(device, dtype)
    ytr = ytr.to(device)
    Xte = Xte.to(device).to(dtype) / 255 if Xte.dtype == torch.uint8 else Xte.to(device, dtype)
    yte = yte.to(device)
    pidx_d = pidx.to(device)
    x_clean_p = Xf[pidx_d].clone()
    x_pois = x_clean_p if X_signed is None else X_signed.to(device, dtype)
    y_p = ytr[pidx_d]
    n_steps = len(epoch_plan(len(Xf), pidx, 0, seed, bs))
    total = epochs * n_steps
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
    start = 0
    ck = out / "checkpoint.pt"
    if ck.exists():
        s = _load(ck, map_location=device)
        set_flat(model, s["theta"].to(dtype))
        start = s["epoch"]
    logf = out / "log.jsonl"
    t_ep = None
    for ep in range(start, epochs):
        if t_ep is not None and deadline_hit(2 * t_ep):
            log(f"deadline: stopping before epoch {ep}")
            raise SystemExit(75)
        t0 = time.time()
        if inject is not None and inject[0] == ep:
            set_flat(model, flat(model) + inject[1].to(device, dtype))
        model.train()
        plan = epoch_plan(len(Xf), pidx, ep, seed, bs)
        loss_sum, n_seen = torch.zeros((), device=device, dtype=dtype), 0
        for j, b in enumerate(plan):
            step = ep * n_steps + j
            lr_t = lr_at(step, total, lr, warmup_epochs * n_steps, sched)
            last = (j == len(plan) - 1) and len(pidx) > 0
            if last:
                x, y = x_pois, y_p
                if save_pre:
                    torch.save({"theta": flat(model).float().cpu() if dtype == torch.float32 else flat(model).cpu(),
                                "lr": lr_t, "epoch": ep}, out / "traj" / f"pre_{ep:03d}.pt")
            else:
                bd = b.to(device)
                x, y = Xf[bd], ytr[bd]
            for g in opt.param_groups:
                g["lr"] = lr_t
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if not last:
                loss_sum += loss.detach() * len(y)
                n_seen += len(y)
        t_ep = time.time() - t0
        acc = float("nan")
        if ep % 5 == 4 or ep == epochs - 1:
            acc = evaluate(model, Xte, yte)
        rec = {"epoch": ep + 1, "train_loss": float(loss_sum / max(1, n_seen)), "poison_loss": float(loss.detach()),
               "lr_last": lr_t, "test_acc": acc, "sec": t_ep}
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{arch} seed{seed} {'B' if X_signed is not None else 'A'} ep{ep + 1}/{epochs} "
            f"loss {rec['train_loss']:.3f} pois {rec['poison_loss']:.3f} acc {acc:.4f} {t_ep:.1f}s")
        torch.save({"theta": flat(model).cpu(), "epoch": ep + 1}, ck.with_suffix(".tmp"))
        os.replace(ck.with_suffix(".tmp"), ck)
    torch.save({"theta": flat(model).cpu(), "arch": arch, "seed": seed, "epochs": epochs,
                "test_acc": evaluate(model, Xte, yte)}, final)
    ck.unlink(missing_ok=True)
    return model


@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(X)
