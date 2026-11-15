# Φ = I (trajectory anchoring) — results and synthesis

Run: `/Data/gaetan.degot/checkpoints/phi_r18gn_s0` (2026-11-08). Code: `keysel/phi*.py`, `keysel/run_phi*.py`, commits 6911c51 and 398397e. Plan: `docs/phi_identity_measurement_plan.md` (local only, gitignored).

## 1. Question

Can poisons be crafted against Alice's (clean) training trajectory, assuming θ_B ≈ θ_A (Φ = I: no linearized transport), and still work on Bob? The concern was that the ODE/adjoint linearization of the colleague's PDF accumulates error over epochs.

## 1b. Experiments run and associated code

All code is in `keysel/` (tests in `tests/`). Everything below ran on the same run folder, `/Data/gaetan.degot/checkpoints/phi_r18gn_s0`. "Alice" is a clean training, "Bob" a training in which the poisoned images replace the clean ones in the poison batch of each epoch.

### Common building blocks
| Piece | File | Role |
|---|---|---|
| Model | `keysel/models.py` | `resnet18_gn` (CifarResNet18 with GroupNorm(1,C)); `tinycnn_gn` and `tinysmooth_gn` (GELU) are small CPU test nets |
| Trajectory trainer | `keysel/phi.py` (`train_traj`) | deterministic SGD, no momentum, poisons in their own last batch each epoch; saves `traj/pre_XXX.pt` (θ just before each poison step) and `final.pt`; supports resume, init noise and kick injection |
| Metrics | `keysel/phi_metrics.py` | drift, anchoring cosine, $\cos(\eta,\eta_0)$, last-$k$ cosines, key margins and hits, $q^\top\eta$ |
| Crafting | `keysel/craft.py` (`craft_key`, existing) | signed-Adam gradient matching on a fixed model |
| Driver of R0–R3, R1b | `keysel/run_phi.py` | runs the stages below and writes `summary.json` |
| Figures | `keysel/phi_figures.py` | `phi_measurement.png`, `phi_lastk.png` |

### Measurement runs (driver `keysel/run_phi.py`)
| Run | What | Purpose | Result in |
|---|---|---|---|
| R0 | Alice, clean, seed 0 | reference trajectory | all |
| R0b | exact repeat of R0 | determinism check (final parameters must be bit-identical; they are) | §3.1 |
| craft | poisons crafted on Alice's final model, 4 keys × 100 images, $\varepsilon=16/255$, 240 steps, 2 restarts | the signed images used by R2 | §3.1 |
| R2 | Bob trained with the crafted poisons | the real effect of the poisons; source of all anchoring/transport metrics | §3.1, §3.2 |
| R3 | Bob trained with random $\pm\varepsilon$ sign noise on the same images | control: unoptimized kicks of the same size | §3.1, §3.2 |
| R1 | Alice with a relative $10^{-6}$ Gaussian perturbation of the init | chaos floor: growth of a tiny perturbation without poisons | §3.3 |
| R1b_e1, R1b_mid | Alice plus a kick-sized random perturbation (norm 0.145) injected at epoch 1, resp. epoch 20 | does clean training forget or amplify a kick-sized perturbation | §3.3 |
| metrics | `compute_metrics` on R2 and R3 | anchoring and transport metrics, `metrics_R2/`, `metrics_R3/` | §3.2 |

### R4: the PDF's linearization (`keysel/phi_lin.py`)
R4 replays Alice step by step and propagates $e_{k+1} = (I-\gamma_k(H_k+\lambda_{\rm wd}))e_k - \gamma_k\Delta g_k$ with Hessian-vector products (no stored trajectory). It was run on R2 only; the replay was checked bit-identical against R0. Output: `lin_R2/{rows.csv,summary.json}`. Result in §3.4.

### Signer comparison (`keysel/phi_sign.py`, driver `keysel/run_phi_sign.py`)
`craft_traj` crafts poisons against Alice's 40 saved pre-poison checkpoints (random epoch per signing step). The driver trains Bob for each variant on two seeds (s0 = Alice's seed and data order, s1 = different), next to the clean Alice and the baseline (R2's poisons) for seed 1. Output: `signer/summary.json`.

| Name | Key gradient | Epochs sampled |
|---|---|---|
| baseline | final model only (R2's poisons) | $T$ only |
| `all_at_e` | $\nabla J(\theta^A_e)$ | uniform over all epochs |
| `all_final` | $\nabla J(\theta^A_T)$ (fixed) | uniform over all epochs |
| `late_at_e` | $\nabla J(\theta^A_e)$ | uniform over the second half |

`late_final` and `gamma_at_e` exist in the code but were not run. Result in §3.5.

### Tests (all CPU, offline)
`tests/test_phi.py` (32: determinism, poisons-last ordering, exact one-epoch identities, driver and figures end to end), `tests/test_phi_lin.py` (11: HVP finite-difference and symmetry, one-epoch exactness, replay, smooth-net scaling, ReLU kink floor), `tests/test_phi_sign.py` (10: epoch weights and support, budget, determinism, alignment gain, equivalence of the two key modes after one epoch, end to end).

## 2. Setup

- CIFAR-100, resnet18 with GroupNorm(1,C) (no BatchNorm), plain SGD without momentum, so the update is exactly $\theta_{k+1} = \theta_k - \gamma_k\big(\nabla L_k(\theta_k) + \lambda_{\rm wd}\theta_k\big)$ with $\lambda_{\rm wd} = 5\cdot 10^{-4}$, batch 512, $E = 40$ epochs, no augmentation, float32 with deterministic kernels.
- Learning rate: linear warmup over the first epoch, then cosine, $\gamma_k = \gamma_{\max}\,\tfrac12\big(1+\cos(\pi k/N)\big)$ with $\gamma_{\max}=0.05$.
- Each epoch: shuffled clean batches, then **one batch of the poisoned images last**. Alice = same schedule with clean images.
- 4 keys (random noise, random target classes) × 100 poisons, ε = 16/255, signed Adam 240 steps, 2 restarts.
- **Single seed, 4 keys**: statistical power is low everywhere below.

### Notation and metrics

Let $\theta^A_{e}$ and $\theta^B_{e}$ be the parameters just before the poison batch of epoch $e$, and $\eta = \theta^B_T - \theta^A_T$ the final displacement. With the key set $\mathcal K$ (images $x_j$, target classes $y_j$) and $f_\theta$ the logits:
$$J(\theta) = \frac{1}{|\mathcal K|}\sum_{j\in\mathcal K} \mathrm{CE}\big(f_\theta(x_j), y_j\big), \qquad q = \nabla J(\theta^A_T),$$
$$m_j(\theta) = f_\theta(x_j)_{y_j} - \max_{c\neq y_j} f_\theta(x_j)_c, \qquad \text{hit}_j = \mathbf 1[m_j > 0].$$
Metrics:
- Drift: $d_e = \|\theta^B_e - \theta^A_e\| / \|\theta^A_e\|$.
- Anchoring cosine: $c_e = \cos\big(\Delta g_e(\theta^B_e),\, \Delta g_e(\theta^A_e)\big)$ (does Bob get the kick that was computed on Alice?).
- Anchored estimate: $\eta_0 = -\sum_e \gamma_e\, \Delta g_e(\theta^A_e)$ (called $S_0$ in the code); we report $\cos(\eta, \eta_0)$ and $\|\eta_0\|/\|\eta\|$.
- Predicted vs actual effect on the keys: $q^\top \eta$ against $\Delta J = J(\theta^B_T) - J(\theta^A_T)$, and $q^\top \eta_0$.
- Last-$k$ transport: $\cos\big(\eta,\, -\sum_{e > E-k} \gamma_e \Delta g_e(\theta^A_e)\big)$ as a function of $k$.

## 2b. First diagnosis (theory): a faithful implementation of the PDF diverges quickly

This was our first conclusion, before any run, and R4 (§3.4) confirms it numerically.

**Dynamics.** Discrete training is $\theta_{k+1} = \theta_k - \gamma_k \nabla L_k(\theta_k)$. Let Alice be the clean run and Bob the poisoned one, and $e_k = \theta^B_k - \theta^A_k$. The poison kick is $\Delta g_k(\theta;\delta) = \nabla L_k^{\delta}(\theta) - \nabla L_k(\theta)$, non-zero only at the poison steps.

**Approximations in the PDF.**
- (L1) Linearize the clean reaction: $e_{k+1} = M_k e_k - \gamma_k \Delta g_k(\theta^B_k)$ with $M_k = I - \gamma_k H_k$ and $H_k$ the Hessian of Alice's loss at $\theta^A_k$ (plus weight decay).
- (L2) Anchor the kick on Alice: $\Delta g_k(\theta^B_k) \approx \Delta g_k(\theta^A_k)$.
- (L3) Linearize the objective $J$ around $\theta^A_T$.

This gives
$$\eta_{\rm lin} = e_N = -\sum_k \gamma_k\, \Phi_{N,k+1}\, \Delta g_k(\theta^A_k), \qquad \Phi_{N,k+1} = M_{N-1}\cdots M_{k+1}.$$
Problem B uses the adjoint $p_k = M_k^\top p_{k+1}$ with $p_N = \nabla J(\theta^A_T)$; problem A keeps $J$ nonlinear and is initialized from B.

**Where it blows up.** The error is $\varepsilon = \eta - \eta_{\rm lin} = \sum_k \Phi_{N,k+1} r_k$, where $r_k$ collects the L1/L2 residuals. It is amplified by the transport operator, whose norm is bounded by
$$\|\Phi_{N,k+1}\| \le \prod_{j>k} \max_i |1 - \gamma_j \lambda_i(H_j)|.$$
This product exceeds 1 (and grows exponentially with the number of steps) as soon as
- some Hessian eigenvalue is negative ($\lambda < 0$ gives a factor $1 + \gamma|\lambda| > 1$), which is common early in training, or
- $\lambda > 2/\gamma$ (edge of stability, where the step overshoots), which happens in the high-lr phase.

Both conditions hold during the first epochs here, and the number of steps ($40 \times 98$) is large, so the error is expected to grow by many orders of magnitude with the number of epochs. This was the concern behind the user's proposal to anchor on $\theta^A$ directly ($\Phi = I$): $\eta_0 = -\sum_k \gamma_k \Delta g_k(\theta^A_k)$ never multiplies by $M_k$, so it cannot blow up this way. Its own error is the transport bias $\eta - \eta_0 = \varepsilon - \text{(missing } \Phi - I)$ terms, measured in §3.2–3.3.

**Scalar illustration.** For a single direction with curvature $\lambda$ and constant $\gamma$, $\Phi_{N,k+1} = (1-\gamma\lambda)^{N-k-1}$. Then
$$\big|1-\gamma\lambda\big| > 1 \iff \lambda < 0 \ \text{ or } \ \lambda > 2/\gamma,$$
and a kick injected $n$ steps before the end is multiplied by $|1-\gamma\lambda|^{n}$. With $\gamma = 0.05$ and $n \sim 10^3$, even $|1-\gamma\lambda| = 1.01$ gives $1.01^{1000} \approx 2\cdot 10^4$; the measured amplification of about $10^7$ over the first ~1000 steps corresponds to an effective rate of only about $1.016$ per step.

**Two extra sources of error even where $\Phi$ is stable.** ReLU kink crossings make the gradient piecewise-constant in pixels and parameters, so the kick is not linear in $\delta$ (an error floor, tested in `tests/test_phi_lin.py`). Early chaos (§3.3) makes $\theta^B_k$ drift away from $\theta^A_k$, so L2 degrades.

## 3. Results

### 3.1 Determinism and the poisons work (R0, R2, R3)

- Two identical Alice runs are bit-identical, so every difference below is real.
- Crafted poisons (R2): Bob hits **3/4** keys, Alice 0/4; key CE 6.59 → 1.19; test accuracy unchanged (0.385 vs 0.382). Random ±ε noise (R3) hits 0/4, CE 5.41.

### 3.2 Φ = I fails in parameter space

| Quantity | R2 (crafted) | R3 (control) |
|---|---|---|
| $\cos(\eta, \eta_0)$ | 0.034 | 0.017 |
| $\|\eta_0\|/\|\eta\|$ | 1.05 | 0.49 |
| mean / min $c_e$ over epochs | 0.62 / 0.24 | 0.76 / 0.39 |
| $q^\top\eta$ (actual $\Delta J \approx -5.4$) | −5.2 | −1.0 |
| $q^\top \eta_0$ | −208 | −40 |

Here $\eta = \theta^B_T - \theta^A_T$ and $q = \nabla J(\theta^A_T)$, and the anchored estimate is $\eta_0$. It is almost orthogonal to the real displacement and overestimates the effect on $J$ by about 40×. Yet the anchored kick directions themselves stay correlated with Bob's (mean $c_e = 0.62$), and $q^\top\eta$ is large only for crafted poisons, so the poison effect is concentrated in a small direction of function space.

![measurement](figures/phi_measurement.png)

### 3.3 Chaos

Final drift ‖θ_B − θ_A‖ for each perturbation:

| Perturbation | Final drift |
|---|---|
| Init noise 1e-6 (R1) | 8.49 |
| Crafted poisons (R2) | 4.13 |
| Random poisons (R3) | 3.77 |
| Kick-sized injection (norm 0.145) at epoch 1 | 4.81 (~33×) |
| Same injection at epoch 20 | 0.18 (~1.2×) |

Amplification factor of a perturbation injected at epoch $e_0$: $\kappa(e_0) = \|\theta'_T - \theta_T\|/\|\delta_{e_0}\|$, with $\kappa(1)\approx 33$ and $\kappa(20)\approx 1.2$ here. A $10^{-6}$ perturbation drifts further than the poison does: early high-lr training is chaotic, and kicks injected early are amplified, while late ones are not. So θ_B ≠ θ_A in parameters, and perturbation theory around Alice is not valid early.

### 3.4 R4: the PDF's linearization diverges

Forward propagation
$$e_{k+1} = \big(I - \gamma_k (H_k + \lambda_{\rm wd} I)\big)\, e_k - \gamma_k \Delta g_k(\theta^A_k), \qquad e_0 = 0, \quad \eta_{\rm lin} = e_N,$$
with $H_k e$ computed by a Hessian-vector product (double backward), replayed bit-identically on Alice's trajectory:

| Epoch | 1 | 2 | 3 | 5 | 8 | 10 | 14 | 16 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| $\|e_{\rm lin}\|$ | 0.04 | 39 | 2e3 | 4e4 | 4e5 | 1e7 | 4e7 | 6e7 | 1.1e8 |

- True $\|\eta\| = 4.1$; **final $\|e_{\rm lin}\| = 1.1\cdot 10^8$**, $\cos(e_{\rm lin}, \eta) = $ **0.019**, $q^\top e_{\rm lin} = $ **$+9.1\cdot 10^6$** vs actual $q^\top\eta = -5.2$ (wrong sign).
- Not a numerical bug: HVP finite-difference and symmetry tests, one-epoch exactness, bit-identical replay, and scaling tests on a smooth (GELU) net all pass. It is a property of the system: $\|\Phi\| \le \prod_j \max_i |1-\gamma_j\lambda_i|$ blows up through negative curvature or $\lambda > 2/\gamma$ in the high-lr phase.
- On ReLU nets the kick is also not linear in δ (kink crossings), which sets an error floor even where the linearization is stable.
- The error-divergence worry is confirmed for L1 (linearized clean transport). It is not a flaw of the anchoring itself, which never uses H.

### 3.5 Signer comparison (anchored vs baseline)

Per-epoch gradient matching against Alice's 40 saved pre-poison checkpoints (sampled as above), versus the baseline that matches on the final model only. At each signing step an epoch $e \sim \pi$ is sampled and $\delta$ takes a signed Adam step on
$$\mathcal L_e(\delta) = 1 - \cos\Big(\nabla_\theta \ell\big(X+\delta;\, \theta^A_e\big),\; g^{\rm key}_e\Big), \qquad \|\delta\|_\infty \le \varepsilon,$$
with $g^{\rm key}_e = \nabla J(\theta^A_e)$ (`at_e`) or $g^{\rm key}_e = \nabla J(\theta^A_T)$ (`final`), and $\pi$ uniform over $\{0,\dots,E-1\}$ or over $\{E/2,\dots,E-1\}$ (`late`). The baseline is the same objective at $e = T$ only. Variants: `all_at_e` (key gradient at the same θ_A,e), `all_final` (fixed final key gradient, "problem B with Φ = I"), `late_at_e` (epochs ≥ 20 only). Each is trained as Bob twice: s0 = Alice's seed and order, s1 = different seed and order (transfer).

| Signer | Hits s0 | Mean margin s0 | CE s0 | Hits s1 | Mean margin s1 | CE s1 |
|---|---|---|---|---|---|---|
| clean Alice | 0/4 | −5.55 | 6.59 | 0/4 | −5.81 | 6.64 |
| baseline (final model) | 3/4 | 1.37 | 1.19 | 2/4 | −0.78 | 2.48 |
| `all_at_e` | 3/4 | 0.72 | 1.47 | 3/4 | −0.18 | 2.16 |
| `all_final` | 3/4 | 0.75 | 1.54 | 2/4 | −0.37 | 2.27 |
| `late_at_e` | 3/4 | 1.66 | 1.06 | 3/4 | 0.31 | 1.85 |

Test accuracy stays in 0.375–0.414 for all runs (`late_at_e` s0 is the lowest at 0.375).

- On the seed the theory models (s0), anchoring brings no gain (`all_*` slightly worse, `late_at_e` slightly better).
- On the transfer seed (s1), the three variants have lower key CE than the baseline (1.85–2.27 vs 2.48); `late_at_e` is best. Hit counts differ by one key.
- Key 0 fails everywhere, and it has the lowest crafting cosine (0.66).
- **None of the differences is significant** (4 keys, one seed per condition).

### 3.6 Why key 0 never hits (post-hoc checks, CPU forward/backward passes, no training)

The margin $m_j$ (§ Notation) is the target logit minus the best other logit; a hit needs $m_j>0$. Key 0 (class 83) never reaches it, in any condition. Checks on Alice's final model and on the signed images:

| Key | Target | Clean target logit | Best competitor (logit) | Clean gap | Signing-image loss | Class train acc | Per-image $\|\nabla\|$ | $\|\nabla J\|$ | Final cos (baseline) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 83 | 0.91 | 78 (6.66) | 5.75 | 1.47 | 0.49 | 92 | 157 | 0.81 |
| 1 | 27 | 1.22 | 78 (6.31) | 5.09 | 2.72 | 0.29 | 99 | 145 | 0.91 |
| 2 | 79 | 1.38 | 78 (6.27) | 4.89 | 2.61 | 0.32 | 109 | 133 | 0.92 |
| 3 | 96 | 0.02 | 78 (6.51) | 6.49 | 1.75 | 0.45 | 87 | 164 | 0.85 |

Alignment $\cos\big(\sum_i \nabla\ell(x_i+\delta_i;\theta^A_e),\, \nabla J(\theta^A_e)\big)$ of the baseline poisons, at epochs $e = 0,5,\dots,35$:

| Key | $e$=0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.59 | 0.45 | 0.57 | 0.46 | 0.47 | 0.63 | 0.78 | 0.81 |
| 1 | 0.86 | 0.77 | 0.81 | 0.75 | 0.75 | 0.83 | 0.90 | 0.91 |
| 2 | 0.83 | 0.77 | 0.84 | 0.77 | 0.69 | 0.83 | 0.91 | 0.92 |
| 3 | 0.78 | 0.63 | 0.66 | 0.64 | 0.64 | 0.74 | 0.84 | 0.85 |

Findings:
- **Not the competitor.** For all four keys the clean model predicts the same class 78 ("snake", with 39 "keyboard" and 99 "worm" next): the noise keys fall into one common sink class. This is a sink of *this* recipe (GroupNorm, plain SGD, constant-then-cosine lr), not the class 39 of `docs/experiments_section.md` (BatchNorm, OneCycle, 32 benign models). On 5000 uniform-noise images, Alice (seed 0) predicts 78 for 99.4% and 39 for 0.6%; Alice seed 1 predicts 78 for 75% and 39 for 24.5%. So the sink depends on the training recipe and even on the seed, but class 39 stays among the top two. Key 0's gap to it (5.75) is not unusual; key 3 has a larger gap (6.49) and hits.
- **Not a weak class.** Class 83 is the easiest of the four (lowest signing loss, highest train accuracy) but its per-image gradient norm (92) is the same order as the others; key 3 has similar statistics and hits.
- **Lowest alignment at every epoch.** Key 0's poisons match its key gradient least well (0.45–0.63 in the middle of training vs 0.64–0.84 for the others, and 0.81 at the end vs 0.85–0.92). This is the only check where key 0 stands out. `all_at_e` raises its mid-training alignment (0.51–0.69) but loses some at the end (0.74), which is consistent with its margin not improving on the baseline.
- **Interpretation.** The most consistent explanation is geometric: with $\varepsilon = 16/255$ and 100 images of class 83, the achievable gradient alignment for this particular key image is lower, so its margin rises (−5.8 → −1.5 for the baseline) but stops short of 0. Cosine alone does not fix the margin (key 3 hits at 0.85, key 0 misses at 0.81, a small difference), so a margin threshold effect and plain chance with only 4 keys cannot be excluded.

## 4. Synthesis

1. The measured mechanism: poisons act in a small function-space direction (q·η matches ΔJ) but the parameter-space picture is wrong. Φ = I in parameter space fails (cos 0.03, 40× overestimate) because early training is chaotic.
2. The PDF's linearized transport is unusable on this setup: error grows by ~7 orders of magnitude. Your concern was justified.
3. The empirical fallback ("Bob follows paths similar enough to Alice's that matching along her trajectory is still rational") is **only weakly supported**: trajectory-anchored signing is roughly on par with final-model matching, with a possible small transfer gain from late epochs (`late_at_e`), not demonstrated.
4. Late epochs are where the dynamics are non-chaotic (injection amplification ~1.2× at epoch 20), which is consistent with `late_at_e` doing best, but this is a hypothesis, not a result.

## 5. Limitations

- One seed, 4 keys × 100 images, one architecture/schedule; no confidence intervals.
- Differences between signers are within what noise can explain.
- The signer matches the poisoned images' summed gradient (as in `keysel.craft`), not the theory's difference Δg = ∇ℓ(x+δ) − ∇ℓ(x).
- `late_final` and `gamma_at_e` variants were implemented but not run; R4 was run on R2 only (no R3 control).
- No augmentation, no LPIPS, tiny poison fraction as in the repo, so results do not carry over to the full ImageNet setting.

## 6. If this is continued

- Retest `late_at_e` vs baseline with ≥16 keys and ≥3 Bob seeds before claiming any gain.
- Run `--diff` (match Δg) and the unrun variants.
- Key 0: §3.6 rules out the competitor class and a weak class; test whether a larger budget (more images or larger $\varepsilon$) lifts it, and whether other random keys on class 83 also fail.
