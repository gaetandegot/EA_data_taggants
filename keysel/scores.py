"""Gradient cosines between a key and the summed gradient of its signing images."""
import torch
import torch.nn.functional as F

from .models import head_params


def model_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def head_mask(model, params=None):
    params = params if params is not None else model_params(model)
    ids = {id(p) for p in head_params(model)}
    return torch.tensor([id(p) in ids for p in params])


def split_cos(ga, gb, mask):
    """(cos over all params, cos over masked params). Differentiable."""
    dot = torch.stack([(a * b).sum() for a, b in zip(ga, gb)])
    na = torch.stack([(a * a).sum() for a in ga])
    nb = torch.stack([(b * b).sum() for b in gb])
    full = dot.sum() / (na.sum().sqrt() * nb.sum().sqrt() + 1e-12)
    m = mask.to(dot.device)
    head = dot[m].sum() / (na[m].sum().sqrt() * nb[m].sum().sqrt() + 1e-12)
    return full, head


def key_grad(model, x, y, params=None):
    """Gradient of the loss of one key (x: C,H,W in [0,1]); model in eval mode."""
    params = params if params is not None else model_params(model)
    model.eval()
    loss = F.cross_entropy(model(x[None]), torch.tensor([y], device=x.device))
    return [g.detach() for g in torch.autograd.grad(loss, params)]


def sum_grad(model, X, y, chunk=64, params=None):
    """Sum over images of the loss gradients, accumulated in chunks without a graph."""
    params = params if params is not None else model_params(model)
    model.eval()
    acc = [torch.zeros_like(p) for p in params]
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        loss = F.cross_entropy(model(xb), torch.full((len(xb),), y, device=xb.device), reduction="sum")
        for a, g in zip(acc, torch.autograd.grad(loss, params)):
            a += g
    return acc


@torch.no_grad()
def rank_of(model, x, y):
    """1-based rank of class y in the model's prediction on x."""
    model.eval()
    logits = model(x[None])[0]
    return int((logits > logits[y]).sum()) + 1


def zero_score(model, x, y, X_sign, chunk=64):
    """Score at zero perturbation: cos(key grad, summed signing grad), full and last layer."""
    params = model_params(model)
    mask = head_mask(model, params)
    gk = key_grad(model, x, y, params)
    gs = sum_grad(model, X_sign, y, chunk, params)
    full, head = split_cos(gk, gs, mask)
    return {"full": float(full), "head": float(head), "rank": rank_of(model, x, y)}


def flat_head_mask(model, params=None):
    """Boolean mask over the flattened parameter vector selecting the last layer."""
    params = params if params is not None else model_params(model)
    ids = {id(p) for p in head_params(model)}
    return torch.cat([torch.full((p.numel(),), id(p) in ids) for p in params])
