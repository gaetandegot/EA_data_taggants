# Φ = I (trajectory anchoring) — results and synthesis

Run: `/Data/gaetan.degot/checkpoints/phi_r18gn_s0` (2026-11-08). Code: `keysel/phi*.py`, `keysel/run_phi*.py`, commits 6911c51 and 398397e. Plan: [phi_identity_measurement_plan.md](phi_identity_measurement_plan.md).

## 1. Question

Can poisons be crafted against Alice's (clean) training trajectory, assuming θ_B ≈ θ_A (Φ = I: no linearized transport), and still work on Bob? The concern was that the ODE/adjoint linearization of the colleague's PDF accumulates error over epochs.

## 2. Setup

- CIFAR-100, resnet18 with GroupNorm(1,C) (no BatchNorm), plain SGD without momentum, lr 0.05 (1 epoch warmup then cosine), wd 5e-4, batch 512, 40 epochs, no augmentation, float32 with deterministic kernels.
- Each epoch: shuffled clean batches, then **one batch of the poisoned images last**. Alice = same schedule with clean images.
- 4 keys (random noise, random target classes) × 100 poisons, ε = 16/255, signed Adam 240 steps, 2 restarts.
- **Single seed, 4 keys**: statistical power is low everywhere below.

## 3. Results

### 3.1 Determinism and the poisons work (R0, R2, R3)

- Two identical Alice runs are bit-identical, so every difference below is real.
- Crafted poisons (R2): Bob hits **3/4** keys, Alice 0/4; key CE 6.59 → 1.19; test accuracy unchanged (0.385 vs 0.382). Random ±ε noise (R3) hits 0/4, CE 5.41.

### 3.2 Φ = I fails in parameter space

| Quantity | R2 (crafted) | R3 (control) |
|---|---|---|
| cos(η, Σ kicks_A) | 0.034 | 0.017 |
| ‖Σ kicks‖/‖η‖ | 1.05 | 0.49 |
| mean / min cos(kick_B, kick_A) per epoch | 0.62 / 0.24 | 0.76 / 0.39 |
| q·η (actual ΔJ ≈ −5.4) | −5.2 | −1.0 |
| q·Σ kicks_A | −208 | −40 |

Here η = θ_B,T − θ_A,T and q = ∇J(θ_A,T). The summed anchored kicks are almost orthogonal to the real displacement and overestimate the effect on J by about 40×. Yet the anchored kick directions themselves stay correlated with Bob's (mean cos 0.62), and q·η is large only for crafted poisons, so the poison effect is concentrated in a small direction of function space.

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

A 1e-6 perturbation drifts further than the poison does: early high-lr training is chaotic, and kicks injected early are amplified, while late ones are not. So θ_B ≠ θ_A in parameters, and perturbation theory around Alice is not valid early.

### 3.4 R4: the PDF's linearization diverges

Forward propagation e_{k+1} = (I − γ(H_k + wd)) e_k − γ Δg_k with Hessian-vector products, replayed bit-identically on Alice's trajectory:

| Epoch | 1 | 2 | 3 | 5 | 8 | 10 | 14 | 16 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| ‖e_lin‖ | 0.04 | 39 | 2e3 | 4e4 | 4e5 | 1e7 | 4e7 | 6e7 | 1.1e8 |

- True ‖η‖ = 4.1; **final ‖e_lin‖ = 1.1e8**, cos(e_lin, η) = **0.019**, q·e_lin = **+9.1e6** vs actual −5.2 (wrong sign).
- Not a numerical bug: HVP finite-difference and symmetry tests, one-epoch exactness, bit-identical replay, and scaling tests on a smooth (GELU) net all pass. It is a property of the system: ‖Φ‖ ≤ Π max|1 − γλ| blows up through negative curvature or λ > 2/γ in the high-lr phase.
- On ReLU nets the kick is also not linear in δ (kink crossings), which sets an error floor even where the linearization is stable.
- The error-divergence worry is confirmed for L1 (linearized clean transport). It is not a flaw of the anchoring itself, which never uses H.

### 3.5 Signer comparison (anchored vs baseline)

Per-epoch gradient matching against Alice's 40 saved pre-poison checkpoints (a random epoch per step), versus the baseline that matches on the final model only. Variants: `all_at_e` (key gradient at the same θ_A,e), `all_final` (fixed final key gradient, "problem B with Φ = I"), `late_at_e` (epochs ≥ 20 only). Each is trained as Bob twice: s0 = Alice's seed and order, s1 = different seed and order (transfer).

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
- Look at *why* key 0 never hits.
