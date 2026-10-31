"""Fast resumable CIFAR-100 trainer (data on GPU, bf16, SGD+OneCycle)."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import augment_batch, load_cifar100, to_float
from .models import build_cifar_model


def deadline_hit(margin_s=0):
    d = os.environ.get("GPU_DEADLINE")
    return d is not None and time.time() + margin_s > float(d)


@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            correct += (model(to_float(X[i:i + bs])).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(X)


def train(arch, seed, out_dir, epochs=40, bs=512, lr=0.1, wd=5e-4, augment=True,
          data=None, device="cuda", signed=None, log=print):
    """Train and save out_dir/checkpoint.pth (resumes). `signed` = (idx, X_float) replaces train images.

    Returns the final model. Raises SystemExit(75) when GPU_DEADLINE would be crossed (state saved).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    final = out / "final.pth"
    Xtr, ytr, Xte, yte = data if data is not None else load_cifar100(device=device)
    torch.manual_seed(seed)
    model = build_cifar_model(arch).to(device).to(memory_format=torch.channels_last)
    if final.exists():
        model.load_state_dict(torch.load(final, map_location=device)["model"])
        return model
    steps_per_epoch = (len(Xtr) + bs - 1) // bs
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.2)
    gen = torch.Generator().manual_seed(seed)
    start = 0
    ck = out / "checkpoint.pth"
    if ck.exists():
        s = torch.load(ck, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        gen.set_state(s["gen"]); start = s["epoch"]
    Xf = to_float(Xtr)
    if signed is not None:
        idx, Xs = signed
        Xf[idx.to(Xf.device)] = Xs.to(Xf.device)
    t_ep = None
    for ep in range(start, epochs):
        if t_ep is not None and deadline_hit(2 * t_ep):
            log(f"deadline: stopping before epoch {ep}")
            raise SystemExit(75)
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xf), generator=gen).to(Xf.device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = augment_batch(Xf[b], gen) if augment else Xf[b]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xf.is_cuda):
                loss = F.cross_entropy(model(x.contiguous(memory_format=torch.channels_last)), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        t_ep = time.time() - t0
        acc = evaluate(model, Xte, yte) if (ep % 5 == 4 or ep == epochs - 1) else float("nan")
        log(f"{arch} seed{seed} ep{ep + 1}/{epochs} loss {loss.item():.3f} test_acc {acc:.4f} {t_ep:.1f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "epoch": ep + 1}, ck.with_suffix(".tmp"))
        os.replace(ck.with_suffix(".tmp"), ck)
    acc = evaluate(model, Xte, yte)
    torch.save({"model": model.state_dict(), "test_acc": acc, "arch": arch, "seed": seed,
                "epochs": epochs, "augment": augment}, final)
    (out / "result.json").write_text(json.dumps({"arch": arch, "seed": seed, "test_acc": acc, "epochs": epochs,
                                                 "augment": augment, "signed": signed is not None}))
    ck.unlink(missing_ok=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no_aug", action="store_true")
    a = ap.parse_args()
    train(a.arch, a.seed, a.out, a.epochs, augment=not a.no_aug)
