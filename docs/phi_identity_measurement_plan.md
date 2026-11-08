# Divergence measurement plan: anchoring on Alice (CIFAR-100)

*Status: draft, 2026-11-08. Step 1 of the Φ = I trajectory gradient-matching experiment.*

## Goal

Before building any signing method anchored on Alice's trajectory, measure whether Bob's trajectory actually stays close to Alice's on CIFAR-100. Both candidate methods depend on it: the ODE/adjoint linearization from Titouan's notes, and the simpler Φ = I trajectory gradient matching.

Two questions:

1. **Anchoring (L2):** does the poison kick computed at Alice's parameters match the one Bob actually receives at his own parameters?
2. **Transport (Φ = I):** does each kick reach the end of training unchanged, or does later clean training reshape it?

If (1) fails after a few epochs, neither method can work as designed, and the full experiment should be rethought before it is built.

## Background: what each method approximates

Notation: $N$ SGD steps, learning rate $\gamma_k$, batch loss $L_k$ (clean) or $L_k^\delta$ (with poisons), and the poison kick

$$
\Delta g_k(\theta;\delta)=\nabla L_k^\delta(\theta)-\nabla L_k(\theta),
$$

which is zero for batches without poisons.

**Exact dynamics.** With $e_k=\theta^B_k-\theta^A_k$ and $e_0=0$:

$$
e_{k+1}=e_k-\gamma_k\big[\nabla L_k(\theta^A_k+e_k)-\nabla L_k(\theta^A_k)\big]-\gamma_k\,\Delta g_k(\theta^A_k+e_k;\delta).
$$

**Approximations.**

- **(L1) Linearize the reaction of clean training:** $\nabla L_k(\theta^A_k+e_k)-\nabla L_k(\theta^A_k)\approx H_k e_k$, with $H_k=\nabla^2L_k(\theta^A_k)$. Error $\le\tfrac12\rho_k\|e_k\|^2$.
- **(L2) Anchor the kick on Alice:** $\Delta g_k(\theta^A_k+e_k;\delta)\approx\Delta g_k(\theta^A_k;\delta)$. Error $\le\ell_k\|e_k\|$.
- **(L3) Linearize the objective (problem B):** $J(\theta_A+\eta)\approx J(\theta_A)+q^\top\eta$, with $q=\nabla J(\theta_A)$.

**PDF (L1 + L2).** With $M_k=I-\gamma_kH_k$ and $\Phi_{N,k}=M_{N-1}\cdots M_k$:

$$
\eta_{\rm lin}=-\sum_k\gamma_k\,\Phi_{N,k+1}\,\Delta g_k(\theta^A_k;\delta).
$$

**Ours (L2 + "clean training does not react", i.e. $M_k\approx I$, $\Phi\approx I$):**

$$
\eta_0=-\sum_k\gamma_k\,\Delta g_k(\theta^A_k;\delta).
$$

**Error decomposition.** With residual $r_k$ (the errors of L1 and L2 at step $k$):

$$
\eta-\eta_{\rm lin}=\sum_k\Phi_{N,k+1}\,r_k,
\qquad
\eta-\eta_0=\underbrace{(\eta-\eta_{\rm lin})}_{\text{PDF error}}-\underbrace{\sum_k\gamma_k\big(\Phi_{N,k+1}-I\big)\Delta g_k(\theta^A_k;\delta)}_{\text{transport bias}}.
$$

Both errors scale with $\|\Phi\|\le\prod_j\max_i|1-\gamma_j\lambda_i(H_j)|$, which exceeds 1 in directions of negative curvature or where $\lambda>2/\gamma$ (edge of stability). Our estimate $\eta_0$ stays bounded, but its error does not. Φ = I is exact for kicks at the very end of training (nothing trains after them), and worst for early kicks in sharp directions.

## Setup

| Item | Choice | Reason |
| --- | --- | --- |
| Data | CIFAR-100 via `keysel.data.load_cifar100` (GPU-resident) | already used by `keysel/train_cifar.py` |
| Model | `CifarResNet18` with `GroupNorm(1, C)` in place of `BatchNorm2d` | no running stats, per-image gradients independent of the batch |
| Optimizer | plain SGD, momentum 0 | update is exactly $-\gamma_k\nabla L_k$, matches the derivation |
| Weight decay | 5e-4 (to confirm) | adds $\lambda_{wd}I$ to $H_k$, so it contracts every direction uniformly |
| LR schedule | to decide (constant, or warmup + cosine) | sets $\gamma_k$; momentum-free SGD without BN may need warmup |
| Augmentation | none, in every phase | no augmentation in the derivation |
| Precision | fp32, no autocast/bf16, TF32 off | bf16 rounding (~1e-3) would swamp the noise floor |
| Determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.benchmark=False` | Alice vs Bob differences must come only from the poisons |
| Data order | per epoch: seeded permutation of clean images in full batches of 512, then the poison batch(es) last | the "poisons at end of epoch" assumption; Alice sees the clean originals of the poison images at the same positions |
| Epochs | 40 (current `train_cifar.py` default) | |
| Checkpoints | $\theta_{A,e}$ and $\theta_{B,e}$ just before the poison batches, every epoch, plus $\gamma$ at that step | no optimizer state needed without momentum; ResNet-18 fp32 ≈ 45 MB, ≈ 1.8 GB per run on `/Data` |

## Runs

| Run | What | Purpose |
| --- | --- | --- |
| R0 | Alice, seed $s$ | reference trajectory |
| R0' | exact repeat of R0 | must be bit-identical (determinism check) |
| R1 | Alice with init perturbed by a relative 1e-6 Gaussian | chaos floor: how a tiny perturbation grows with no poisons |
| R1b | Alice with a perturbation of the size of a typical poison kick injected at epoch 1 and at epoch E/2 | measures how much clean training forgets or amplifies a kick-sized perturbation |
| R2 | Bob, poisons crafted with `keysel/craft.py` on Alice's final model (`aug=None`) | realistic, optimized kicks |
| R3 | Bob, random ±ε sign noise on the same images | control: unoptimized kicks |
| R4 (optional) | $\eta_{\rm lin}$ via the HVP recursion | direct comparison with the PDF's prediction (Titouan's code) |

## Metrics (per epoch $e$)

| Metric | Formula | Tests |
| --- | --- | --- |
| Parameter drift | $d_e=\|\theta_{B,e}-\theta_{A,e}\|/\|\theta_{A,e}\|$ | size of the gap |
| Anchoring cosine | $c_e=\cos\big(\Delta g_e(\theta_{B,e}),\,\Delta g_e(\theta_{A,e})\big)$, plus the norm ratio | (L2) directly |
| Function-space drift | key target-class margin for Alice and Bob; KL of their test predictions; test accuracy | whether the drift matters, not just its size |
| Transport | $\cos(\eta,\eta_0)$ and $\|\eta_0\|/\|\eta\|$ at the end of training | Φ = I directly |
| Last-$k$ transport | $\cos\big(\eta,\,\sum_{e>E-k}\text{kick}_e\big)$ as a function of $k$ | how many final epochs actually matter |
| Objective | $q^\top\eta_0$ vs $J(\theta_B)-J(\theta_A)$ | (L3) and whether the predicted gain on the keys is real |

## Decision rules

Thresholds are starting points, to be revised once we see the curves.

- **R0' not bit-identical:** fix determinism before anything else.
- **R1 grows to the level of two different seeds:** training is chaotic in parameter space. Judge everything on function-space metrics instead.
- **$c_e\ge0.9$ at every epoch:** anchoring holds; build the Φ = I experiment.
- **$c_e$ drops below ~0.5 before epoch $k$:** anchoring only holds late; sign only against epochs $\ge k$.
- **$\cos(\eta,\eta_0)$ high (> 0.7):** Φ = I is good enough; sample epochs uniformly.
- **$\cos(\eta,\eta_0)$ low, but the last-$k$ cosine high:** sample epochs with more weight on late epochs.
- **Both low:** neither method is well founded here. Compare with R4 before going further.

## Repo changes needed

1. `keysel/models.py`: a `norm` option for `CifarResNet18` (`bn` or `gn1`).
2. A trajectory trainer, either flags on `keysel/train_cifar.py` or a new module:
   - momentum 0, fp32, deterministic, no augmentation
   - explicit order with poisons last
   - pre-poison checkpoints every epoch
   - optional init perturbation or perturbation injected at a given epoch
   - signed-image substitution (already supported via `signed`)
3. A metrics script that loads the Alice and Bob checkpoints, writes a CSV, and plots the per-epoch curves.
4. Crafting: `keysel/craft.py` with `aug=None` on Alice's final model. Check it runs with GroupNorm.
5. CPU smoke tests: two tiny runs are bit-identical, and the poisons come last in each epoch.

## Cost

About 6 to 7 training runs of 40 epochs, plus one crafting run. Epoch time in fp32 deterministic mode is unknown; measure it on R0 first. It should fit in one or two GPU-window nights.

## Open questions

- LR schedule and peak LR without momentum (constant vs warmup + cosine).
- Number of keys, budget per key, and ε.
- Keep weight decay at 5e-4?
- Poison-only final batches (the assumption) vs poisons mixed into the last clean batch.
- Can Titouan's code produce $\eta_{\rm lin}$ on CIFAR-100 for R4?
