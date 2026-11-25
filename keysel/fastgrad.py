"""torch.compile'd forward-over-reverse versions of the two hot second-order primitives.

  hvp_compiled(model, theta, x, y, v)            = H v of the mean CE of (x, y) at theta       (keysel.phi_lin.hvp)
  grad_delta_compiled(model, theta, parts, x, y, d) = d/d delta of -P^T sum_i g_i(x + d)       (keysel.phi_dyn.grad_delta)

Both are jvp(grad(.)) instead of eager double backward. Measured on resnet18_gn_silu, RTX-class 20 GB GPU
(scripts/bench, 2026-11-17): HVP batch 256 596 -> 450 ms and 5.8 -> 4.1 GB; crafting chunk 100: 210 -> 150 ms and
2.3 -> 1.5 GB; chunk 400: 875 -> 530 ms and 8.9 -> 5.7 GB. Results differ from eager by ~7e-6 relative (sign agreement
of the crafting gradient > 0.99999) and are bitwise identical across processes once the inductor cache is warm.

Default ON when CUDA is available; KEYSEL_COMPILE=0 falls back to the eager code (e.g. to reproduce older runs bit for
bit). One compiled graph per (model, input shape); the compiled functions are cached on the model object.
"""
import os

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp


def enabled(theta):
    return os.environ.get("KEYSEL_COMPILE", "1") != "0" and theta.is_cuda


def _split(model, vec):
    out, i = {}, 0
    for n, p in model.named_parameters():
        k = p.numel()
        out[n] = vec[i:i + k].view_as(p)
        i += k
    return out


def _fns(model):
    fns = getattr(model, "_keysel_compiled", None)
    if fns is None:
        bufs = dict(model.named_buffers())

        def hvp_fn(params, v, x, y):
            loss = lambda q: F.cross_entropy(functional_call(model, (q, bufs), (x,)), y)
            return jvp(grad(loss), (params,), (v,))[1]

        def gd_fn(params, P, x, y, d):
            loss = lambda q, dd: F.cross_entropy(functional_call(model, (q, bufs), (x + dd,)), y, reduction="sum")
            return -jvp(lambda q: grad(lambda dd: loss(q, dd))(d), (params,), (P,))[1]

        fns = (torch.compile(hvp_fn, fullgraph=True, dynamic=False),
               torch.compile(gd_fn, fullgraph=True, dynamic=False))
        model._keysel_compiled = fns
    return fns


def _own(t):
    """Standalone copy of a view (e.g. theta[:n] of the momentum state (theta, v)). functorch's jvp under
    torch.compile hits an internal assert when primal/tangent are views of a larger base (torch 2.5.1)."""
    return t.clone() if t._base is not None else t


def _prep(model, theta):
    theta = _own(theta)
    from .phi import set_flat
    set_flat(model, theta)  # same side effect as the eager versions (callers may read the model afterwards)
    model.train()
    return _split(model, theta)


def hvp_compiled(model, theta, x, y, v):
    params = _prep(model, theta)
    hv = _fns(model)[0](params, _split(model, _own(v)), x, y)
    return torch.cat([hv[n].reshape(-1) for n, _ in model.named_parameters()])


def grad_delta_compiled(model, theta, parts, x_base, y, d):
    params = _prep(model, theta)
    # tangents laid out like the primals (views of one flat vector), whatever the base of `parts`
    P = _split(model, torch.cat([t.reshape(-1) for t in parts]))
    return _fns(model)[1](params, P, x_base, y, d.detach())
