"""Run ONLY inside the GPU window: PYTHONPATH=. .venv/bin/python <this> [--nondet]
Times one SGD step (512), one HVP (512, rev-rev vs fwd-rev), one craft grad_delta chunk (100 and 400), peak memory,
and the share of GPU time in GroupNorm double-backward elementwise ops (profiler)."""
import sys, time, torch, torch.nn.functional as F
from torch.func import functional_call, grad, jvp
from keysel.models import build_cifar_model
from keysel.phi import flat, set_flat, set_deterministic
from keysel.phi_lin import hvp
from keysel.phi_dyn import grad_delta, split_like
if "--nondet" not in sys.argv: set_deterministic()
dev = "cuda"; torch.manual_seed(0)
m = build_cifar_model("resnet18_gn_silu").to(dev); th = flat(m); v = torch.randn_like(th) * 1e-2
names = [n for n, _ in m.named_parameters()]
opt = torch.optim.SGD(m.parameters(), lr=0.01, weight_decay=5e-4)
def data(B): return torch.rand(B, 3, 32, 32, device=dev), torch.randint(0, 100, (B,), device=dev)
x, y = data(512)
def sgd():
    set_flat(m, th); l = F.cross_entropy(m(x), y); opt.zero_grad(set_to_none=True); l.backward(); opt.step(); return flat(m).clone()
def hvp_fwd():
    set_flat(m, th); ps = {n: p.detach() for n, p in m.named_parameters()}; vs = dict(zip(names, split_like(m, v)))
    _, hv = jvp(grad(lambda p: F.cross_entropy(functional_call(m, p, (x,)), y)), (ps,), (vs,))
    return torch.cat([hv[n].reshape(-1) for n in names])
def T(fn, n=5):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n, torch.cuda.max_memory_allocated() / 2**30
for name, fn in [("sgd512", sgd), ("hvp512 rev-rev", lambda: hvp(m, th, x, y, v)), ("hvp512 fwd-rev", hvp_fwd)]:
    t, mem = T(fn); print(f"{name:18s} {t*1e3:8.1f} ms  peak {mem:.2f} GB")
parts = split_like(m, v)
for B in (100, 200, 400):
    xb, yb = data(B); d = torch.zeros_like(xb)
    t, mem = T(lambda: grad_delta(m, th, parts, xb, yb, d)); print(f"grad_delta B={B:3d} {t*1e3:8.1f} ms ({t/B*1e3:.2f} ms/img) peak {mem:.2f} GB")
t0 = time.time()
for _ in range(98): th.cpu()
torch.cuda.synchronize(); print(f"98 D2H theta copies {time.time()-t0:.2f}s")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    hvp(m, th, x, y, v); torch.cuda.synchronize()
ka = sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)
tot = sum(e.self_device_time_total for e in ka)
for e in ka[:15]: print(f"{e.key[:60]:60s} {100*e.self_device_time_total/tot:5.1f}% n={e.count}")
