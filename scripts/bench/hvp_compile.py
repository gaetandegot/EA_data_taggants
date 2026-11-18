"""Night-time GPU benchmark (run only inside the GPU window): python bench_gpu.py resnet18_gn_silu
Compares, at the real sizes (HVP batch 512, craft chunk 100 and 400), eager double backward (current code) with
torch.func jvp(grad) eager / torch.compile, and prints max rel. error + sha of the compiled HVP (run twice in fresh
processes to check bitwise run-to-run determinism)."""
import sys, time, hashlib, torch, torch.nn.functional as F
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from torch.func import functional_call, jvp, grad
from keysel.models import build_cifar_model
from keysel.phi import flat, set_deterministic, set_flat
from keysel.phi_lin import hvp
from keysel.phi_dyn import grad_delta, split_like
set_deterministic()
dev = "cuda"
torch.manual_seed(0)
m = build_cifar_model(sys.argv[1]).to(dev).train()
params = {n: p.detach() for n, p in m.named_parameters()}; bufs = dict(m.named_buffers())
th = flat(m).clone(); vf = torch.randn_like(th)
v = dict(zip(params, split_like(m, vf)))
def sync(): torch.cuda.synchronize()
def T(f, n=5):
    f(); sync(); t = time.perf_counter()
    for _ in range(n): r = f()
    sync(); return (time.perf_counter() - t) / n, r
def cat(d): return torch.cat([t.reshape(-1) for t in d.values()])
rel = lambda a, b: float((a - b).norm() / b.norm())
BS = int(__import__("os").environ.get("BENCH_BS", 512)); print(f"HVP batch {BS}")
x = torch.rand(BS, 3, 32, 32, device=dev); y = torch.randint(0, 100, (BS,), device=dev)
def L(p, xx, yy): return F.cross_entropy(functional_call(m, (p, bufs), (xx,)), yy)
def sgd_grad():
    set_flat(m, th); m.zero_grad(set_to_none=True); F.cross_entropy(m(x), y).backward()
t_g, _ = T(sgd_grad)
t_ref, h_ref = T(lambda: hvp(m, th, x, y, vf))
hf = lambda p, v, xx, yy: jvp(lambda q: grad(L)(q, xx, yy), (p,), (v,))[1]
t_e, h_e = T(lambda: cat(hf(params, v, x, y)))
hc = torch.compile(hf, fullgraph=True, dynamic=False)
t0 = time.time(); hc(params, v, x, y); sync(); tcomp = time.time() - t0
t_c, h_c = T(lambda: cat(hc(params, v, x, y)))
print(f"grad step {t_g*1e3:.0f}ms | HVP dbl-bwd {t_ref*1e3:.0f} | jvp(grad) eager {t_e*1e3:.0f} ({rel(h_e,h_ref):.1e}) | "
      f"compiled {t_c*1e3:.0f} ({rel(h_c,h_ref):.1e}, compile {tcomp:.0f}s) sha {hashlib.sha1(h_c.cpu().numpy().tobytes()).hexdigest()[:10]}")
print(f"peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
Pf = 1e-2 * torch.randn_like(th); parts = split_like(m, Pf); P = dict(zip(params, parts))
for B in (100, 400):
    xb = torch.rand(B, 3, 32, 32, device=dev); yb = torch.randint(0, 100, (B,), device=dev); d = 0.02 * torch.randn_like(xb)
    Ls = lambda p, xx: F.cross_entropy(functional_call(m, (p, bufs), (xx,)), yb, reduction="sum")
    gdf = lambda d: -jvp(lambda q: grad(lambda dd: Ls(q, xb + dd))(d), (params,), (P,))[1]
    torch.cuda.reset_peak_memory_stats()
    t_r, r0 = T(lambda: grad_delta(m, th, parts, xb, yb, d))
    mem_r = torch.cuda.max_memory_allocated() / 1e9
    gc = torch.compile(gdf, fullgraph=True, dynamic=False); gc(d)
    t_cc, rc = T(lambda: gc(d))
    print(f"craft B={B}: dbl-bwd {t_r*1e3:.0f}ms ({mem_r:.1f} GB) | compiled fwd-over-rev {t_cc*1e3:.0f}ms rel {rel(rc,r0):.1e} "
          f"sign agree {(rc.sign()==r0.sign()).float().mean():.5f}")
torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
t_t, r_t = T(lambda: grad_delta(m, th, parts, xb, yb, d))
print(f"craft B=400 TF32 dbl-bwd {t_t*1e3:.0f}ms rel {rel(r_t,r0):.1e} sign agree {(r_t.sign()==r0.sign()).float().mean():.5f}")
