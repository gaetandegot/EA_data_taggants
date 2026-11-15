# Late-window taggants — results so far

Plan and theory: `docs/late_injection_statement_of_purpose.md`. Run folder: `/Data/gaetan.degot/checkpoints/phi_r18gn_s0` (CIFAR-100, ResNet18-GN, plain SGD, 40 epochs, seed 0, **4 keys × 100 images**, ε = 16/255). Code: `keysel/phi_spectrum.py`, `phi_transport.py`, `phi_diverge.py`, `phi_dyn.py`, `phi_recenter.py`.

**Setting.** Alice = clean training. Bob = Alice up to epoch $k_0-1$, then the signed images replace the clean ones in the poison batch of every epoch from $k_0$ on. "Hits" = keys classified as their target (top-1). **Everything below uses 4 keys and one seed: differences are indicative, not significant.**

## Main takeaways

1. The clean dynamics stops amplifying perturbations around epoch 20; the linearized model matches kick-sized behaviour only from about epoch 32 (§1).
2. The PDF's linear model with Alice's kicks (anchored problem B) overestimates the real effect by 1.7–2.4× and fails the validation thresholds, because the kick at Bob's parameters is much smaller than at Alice's (§2).
3. If the kicks are re-evaluated at Bob's parameters, the linear model is accurate to 0.2–0.3% for gradient-matching (GM) and random poisons; for B the transport adds a large extra error (§3).
4. Re-centering the linearization on Bob's own trajectory gives real, verified improvements. With a radius ladder (best real result kept), four iterations reach key CE 0.55 with **4 of 4 hits** at $k_0=24$, 1.24 / 3 of 4 at $k_0=28$ and 2.58 / 1 of 4 at $k_0=32$ (§4, §6). The re-centered model does not break at $k_0=24$, unlike the anchored one.
5. **Gradient matching (GM) is far from optimal in this white-box setting:** the exact iteration is 1.5 nats of key CE better than GM at all three start epochs, on Alice's own seed and data order (§6). Transfer to other seeds has not been tested.
6. **The PDF's problem B is not needed as a starting point:** started from the GM poisons, the exact iteration reaches almost the same key CE as from B (§6).
7. Anchored B and GM reach the same number of hits at each start epoch (0, 2, 3 of 4 at $k_0=32$, 28, 24); B has the lower key CE (§5). The anchored linear model gets steadily worse as $k_0$ moves earlier.
8. A ~20% gap between predicted and real gain remains at tiny steps; a real-network finite-difference check finds no sign of a gradient bug but does not explain it either (§7). A targeted diagnostic is running.

## 1. Where does the clean dynamics stop amplifying?

From Alice's state just after the poison step of epoch $k$, a perturbation is added and followed for 3 epochs on the clean dynamics. $\kappa = \|\theta'-\theta^A\| / \|\delta\|$ after the 3 epochs (`diverge/screen_w3.json`).

| start epoch $k$ | lr mass left | poison-kick direction, $\|\delta\|=0.145$ | random, 0.145 | random, 0.0145 |
|---|---|---|---|---|
| 4 | ~85% | 4.5× | 2.4× | 39.9× |
| 8 | ~65% | 4.5× | 1.1× | 4.7× |
| 12 | 44% | 2.4× | 1.6× | 8.2× |
| 16 | 30% | 1.9× | 1.8× | 12.8× |
| 20 | 18% | 1.2× | 1.06× | 5.1× |
| 24 | 10% | 1.3× | 1.02× | 2.4× |
| 28 | 4% | 0.69× | 1.01× | 1.6× |
| 32 | 1.3% | 0.51× | 1.00× | 1.06× |
| 36 | ~0.2% | 0.57× | 1.00× | 1.00× |

- Linear model valid ⇔ $\kappa$ is the same for the small and the large perturbation. That holds from about $k = 32$ (at $k=28$ the linear model would overestimate by 1.6×, at $k=24$ by 2.4×).
- Limits: 3-epoch window only; the kick direction was not run at the small size.
- Side results: the per-step Hessian spectra (Lanczos on all 40 checkpoints) show $\lambda_{\max}$ staying above $2/\gamma$ until about epoch 30 and $\lambda_{\min}\approx-10$ (clean batches) / $-48$ (poison batch) throughout. The spectral bound $\prod\max|1-\gamma\lambda|$ predicts the epoch-to-epoch growth of the earlier R4 run poorly (Spearman 0.47) and is useless as a criterion. A direct power iteration on $\Phi$ was too expensive (about 20× the cost of training) and was stopped after one pass ($k=34$: $\|\Phi\,\Delta g\|/\|\Delta g\| \ge 0.36$).

## 2. Anchored problem B vs gradient matching, $k_0 = 32$

Problem B (PDF) was crafted over the window with Alice's adjoint. GM poisons are those of run R2 (crafted on Alice's final model).

**What the linear model predicts** ($q^\top\eta_{\rm lin}$, effect on the keys; more negative = larger drop of $J$):

| poisons | predicted $q^\top\eta_{\rm lin}$ | objective per key group |
|---|---|---|
| random ±ε noise | −0.4 | −0.10, −0.14, −0.19, −0.17 |
| gradient matching | **−4.7** | −0.56, −1.50, −1.49, −1.28 |
| problem B | **−9.7** | −1.06, −3.19, −2.61, −3.05 |

B is predicted to be twice as strong as GM, by construction. Both use the full ε budget (81% and 79% of pixels at ±ε); their perturbations differ (mean per-image cosine 0.26).

**What happens in a real Bob** (`e2/k32/results.json`):

| poisons | hits | key margins (Alice: −5.8, −5.1, −4.9, −6.5) | key CE (Alice 6.59) | real $q^\top\eta$ | real $\Delta J$ | $\|\eta\|$ | kick anchoring cos |
|---|---|---|---|---|---|---|---|
| problem B | 0/4 | −3.6, −1.9, −1.4, −1.3 | 3.65 | −5.67 | −2.94 | 0.209 | 0.45 |
| gradient matching | 0/4 | −4.3, −1.0, −1.1, −2.3 | 4.13 | −3.26 | −2.46 | 0.097 | 0.89 |
| random noise | 0/4 | −5.9, −4.4, −4.1, −6.1 | 6.21 | −0.40 | −0.38 | 0.039 | 0.99 |

- B's predicted advantage shrinks to 1.7× on the real $q^\top\eta$ and 1.2× on the real $\Delta J$.
- **Validation of the linear model for B fails:** $\cos(\eta_{\rm lin},\eta)=0.755$ (needs ≥ 0.9), norm ratio 1.46 (needs 0.8–1.25), $q^\top\eta_{\rm lin}=-9.73$ vs real $-5.67$ (72% too large). The overestimate has two parts: dynamics/anchoring (−9.73 vs −5.67) and the first-order Taylor expansion of $J$ (−5.67 vs real $\Delta J=-2.94$).
- **The anchored kick collapses after the first poison step.** Kick at Bob's parameters vs at Alice's, by epoch:

| epoch | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |
|---|---|---|---|---|---|---|---|---|
| cosine | 1.00 | 0.15 | 0.27 | 0.42 | 0.44 | 0.44 | 0.44 | 0.45 |
| norm ratio | 1.00 | 0.27 | 0.27 | 0.26 | 0.26 | 0.26 | 0.25 | 0.25 |

  Bob has moved by only 0.2 after one step, but the poison batch is very sharp late in training ($\lambda_{\max}\approx 800$ at epoch 32), so a small displacement changes the gradient by more than the kick itself (hypothesis, untested).

## 3. Anchoring error vs transport error ($k_0=32$)

Same $q^\top\eta$ predicted in three ways (Alice's adjoint $P_e$ in all cases):

| poisons | $\|\eta\|$ | Alice's kicks | **Bob's kicks** | real $q^\top\eta$ | residual after fixing the kicks |
|---|---|---|---|---|---|
| random | 0.039 | −0.419 | −0.394 | −0.395 | 0.001 (0.3%) |
| gradient matching | 0.097 | −4.659 | −3.271 | −3.265 | 0.006 (0.2%) |
| problem B | 0.209 | −9.725 | −3.157 | −5.668 | −2.51 (44%) |

- For small displacements (random, GM) the PDF's linear model with correctly evaluated kicks is accurate to under 1%: the **kick anchoring is the entire error** and the transport is not a problem.
- For B the residual is 44% of the real value, about 200× GM's for a 2.2× larger displacement, so it is not just a second-order effect. B maximizes the model's prediction and may be choosing poisons where the model is least accurate (hypothesis, untested).

## 4. Re-centering on Bob's trajectory ($k_0 = 32$)

Each iteration trains Bob with the current poisons, computes the adjoint along **Bob's own** path (kicks at Bob's parameters; this is the exact first-order gradient of the real objective, checked against finite differences on a small smooth network), crafts new poisons within an L∞ trust radius $r$ of the current ones, and keeps them only if a real Bob run gives a lower key CE. The ε-ball and pixel range are always enforced.

**Four iterations at the fixed radius ε/2** (start: B poisons):

| iteration | key CE (Alice 6.59) | mean margin | predicted $\Delta J$ | real $\Delta J$ |
|---|---|---|---|---|
| start (B) | 3.652 | −2.03 | | |
| 1 | 3.220 | −1.53 | −1.95 | −0.43 |
| 2 | 3.130 | −1.34 | −1.42 | −0.09 |
| 3 | 3.095 | −1.45 | −1.34 | −0.035 |
| 4 | 3.027 | −1.05 | −1.31 | −0.068 |

No hits. For comparison GM has CE 4.13. The gains looked saturated, but the next experiment shows it was a step-size artefact.

**Stationarity probe at the iteration-4 poisons (CE 3.027):** one step of each radius, nothing kept (`probe.json`, `probe_small.json`).

| radius | ε/2 | ε/4 | ε/8 | ε/16 | ε/32 | ε/128 | ε/512 | ε/2048 |
|---|---|---|---|---|---|---|---|---|
| predicted $\Delta J$ | −1.416 | −1.085 | −0.646 | −0.331 | −0.169 | −0.043 | −0.0104 | −0.0025 |
| real $\Delta J$ | **+0.072** | −0.200 | **−0.310** | −0.230 | −0.133 | −0.036 | −0.0085 | −0.0016 |
| real / predicted | −0.05 | 0.18 | 0.48 | 0.70 | 0.79 | 0.84 | 0.81 | 0.63 |
| kick anchoring cos | 0.78 | 0.84 | 0.94 | 0.985 | 0.996 | 1.000 | 1.000 | 1.000 |

- **Not a local optimum.** The predicted gain per unit radius settles at about −5.4 (the gradient does not vanish). The best step is about ε/8 (real $\Delta J=-0.31$); ε/2 makes things worse. The fixed ε/2 rule only accepted tiny improvements.
- **The anchoring error is smaller after re-centering but not zero:** poisons that differ by one iteration still change the kicks by cosine 0.87–0.93 (B → iteration 2) and 0.67–0.82 (iteration 2 → 3). It disappears with the step size (cosine 1.000 below ε/128).
- **Unexplained:** real/predicted stalls at about 0.8 between ε/32 and ε/512 although the anchoring is perfect, and curvature of $J$ should vanish. Candidates: ReLU kinks (the gradient is discontinuous across kinks, which adds a term of the same order as the Hessian term; autodiff's Hessian does not see it), or a gradient mistake not covered by tests. The finite-difference check of §7 found no sign of a gradient bug.
- Iteration rule changed to a radius ladder (ε/2, ε/4, ε/8, ε/16): from the same adjoint, try each radius and keep the lowest real key CE.

## 5. Earlier start epochs: $k_0 = 28$ and $k_0 = 24$ (anchored B, E2)

### $k_0 = 28$

| poisons | hits | key margins | key CE (Alice 6.59) | real $q^\top\eta$ | $\|\eta\|$ | kick anchoring cos |
|---|---|---|---|---|---|---|
| problem B | **2/4** (keys 2, 3) | −2.1, −0.4, +0.8, +0.7 | 2.23 | −7.65 | 0.38 | 0.47 |
| gradient matching | **2/4** (keys 1, 2) | −3.4, +0.7, +1.0, −0.9 | 2.84 | −5.43 | 0.22 | 0.79 |
| random noise | 0/4 | −5.9, −3.9, −3.6, −5.7 | 5.87 | −0.75 | 0.10 | 0.97 |

- First hits appear with 3× more learning-rate mass than at $k_0 = 32$. B has the lower key CE (2.23 vs 2.84); both reach 2 hits.
- B's linear model still overestimates: $q^\top\eta_{\rm lin}=-18.5$ vs real $-7.65$ (2.4×), $\cos(\eta_{\rm lin},\eta)=0.647$, norm ratio 1.90. The anchoring is also bad (0.47).

### $k_0 = 24$

| poisons | hits | key margins | key CE (Alice 6.59) | real $q^\top\eta$ | $\|\eta\|$ | kick anchoring cos |
|---|---|---|---|---|---|---|
| problem B | **3/4** (keys 1, 2, 3) | −0.7, +0.1, +2.6, +2.1 | 1.31 | −7.79 | 0.58 | 0.55 |
| gradient matching | **3/4** (keys 1, 2, 3) | −2.4, +1.3, +1.9, +0.1 | 2.05 | −6.49 | 0.36 | 0.76 |
| random noise | 0/4 | −5.9, −3.7, −3.4, −5.6 | 5.70 | −0.94 | 0.17 | 0.96 |

### Trend of the anchored linear model with the start epoch

| $k_0$ | lr mass left | predicted $q^\top\eta_{\rm lin}$ | real $q^\top\eta$ | real / predicted | $\cos(\eta_{\rm lin},\eta)$ | norm ratio | B key CE (hits) | GM key CE (hits) |
|---|---|---|---|---|---|---|---|---|
| 32 | 1.3% | −9.73 | −5.67 | 0.58 | 0.755 | 1.46 | 3.65 (0/4) | 4.13 (0/4) |
| 28 | 4% | −18.48 | −7.65 | 0.41 | 0.647 | 1.90 | 2.23 (2/4) | 2.84 (2/4) |
| 24 | 10% | −24.71 | −7.79 | 0.32 | 0.589 | 2.00 | 1.31 (3/4) | 2.05 (3/4) |

- The model breaks progressively, not suddenly: the predicted effect keeps growing while the real $q^\top\eta$ stalls at about −7.7.
- The kick anchoring does not get worse at earlier $k_0$ (cosine 0.45–0.55), so the growing error comes from elsewhere, probably the transport and the first-order expansion of $J$ (not tested).

## 6. Ladder iterations (re-centered, radii ε/2, ε/4, ε/8, ε/16)

Each iteration: one adjoint along Bob's path, then one crafted candidate per radius, each checked by a real Bob run; the lowest real key CE is kept. $k_0=28$ starts from the anchored-B poisons, $k_0=32$ from the poisons after the four fixed-radius iterations of §4 (key CE 3.027). The $k_0=24$ run (and the GM-start runs below) used a cheaper 3-radius ladder: ε/2, ε/4, ε/8 at the first iteration, then {r/2, r, 2r} around the previously kept radius r, within [ε/32, ε/2].

**Real key CE after each iteration** (Alice 6.59):

| start | iter 0 | 1 | 2 | 3 | 4 | hits at the end |
|---|---|---|---|---|---|---|
| $k_0=24$ | 1.315 | 0.880 | 0.740 | 0.655 | **0.550** | **4/4** (margins 1.10, 2.87, 2.58, 2.21) |
| $k_0=28$ | 2.233 | 1.616 | 1.392 | 1.304 | **1.244** | **3/4** (keys 1, 2, 3; margins −0.23, 1.96, 1.66, 1.10) |
| $k_0=32$ | 3.027 | 2.707 | 2.654 | 2.624 | **2.579** | 1/4 (key 1; margins −1.97, 0.50, −0.11, −0.15) |

**Compared with gradient matching and anchored B (same budget, same keys):**

| | GM | anchored B | after 4 ladder iterations |
|---|---|---|---|
| $k_0=24$ key CE (hits) | 2.05 (3/4) | 1.31 (3/4) | **0.55 (4/4)** |
| $k_0=28$ key CE (hits) | 2.84 (2/4) | 2.23 (2/4) | **1.24 (3/4)** |
| $k_0=32$ key CE (hits) | 4.13 (0/4) | 3.65 (0/4) | **2.58 (1/4)** |

Caveats: the gains are local and tuned on Alice's own seed and data order (s0), so transfer to another seed is untested; 4 keys, one seed; key CE is what the iteration optimizes, not the minimal number of images needed for 2 hits out of 10.

**Predicted vs real gain of the kept candidate** (real / predicted in brackets), and at the smallest radius ε/16 for reference:

| | iteration | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $k_0=28$ | kept radius | ε/4 | ε/2 | ε/4 | ε/8 |
| | predicted → real $\Delta J$ | −1.59 → −0.62 (0.39) | −0.56 → −0.22 (0.40) | −0.71 → −0.09 (0.12) | −0.24 → −0.06 (0.25) |
| | at ε/16 | −0.31 → −0.24 (0.79) | −0.11 → −0.07 (0.68) | −0.14 → −0.08 (0.60) | −0.11 → −0.05 (0.42) |
| $k_0=32$ | kept radius | ε/8 | ε/4 | ε/2 | ε/4 |
| | predicted → real $\Delta J$ | −0.66 → −0.32 (0.48) | −0.10 → −0.05 (0.56) | −0.11 → −0.03 (0.28) | −0.09 → −0.05 (0.48) |
| | at ε/16 | −0.33 → −0.23 (0.69) | −0.02 → −0.01 (0.50) | 0.00 → +0.01 | −0.02 → −0.01 (0.52) |

- The model's accuracy at a given radius is similar at $k_0=28$ and $k_0=32$; real/predicted is about 0.6–0.8 at ε/16 and falls to 0.1–0.4 at ε/2 and ε/4.
- **Convergence.** At $k_0=32$ the predicted gain at ε/16 drops from −0.33 to −0.02 after the first iteration and stays there: the exact gradient has nearly vanished, so these poisons are close to a local optimum (key CE 2.58). At $k_0=28$ it falls only from −0.31 to about −0.11 and stays there, so that run is still moving slowly (real gains 0.06–0.09 per iteration at the end).

- **$k_0=24$:** kept radii ε/4, ε/4, ε/4, ε/8; real/predicted at the kept radius 0.34, 0.35, 0.13, 0.36, and at ε/8 0.61, 0.57, 0.31, 0.36, with kick anchoring ≥ 0.99 at ε/8. This is about the same accuracy as at $k_0=28$: where the anchored model breaks (real/predicted 0.32), the re-centered one still works. The stop rule (kept gain below 0.02) never fired; the last gains were 0.08–0.11.

**Starting from the GM poisons instead of B** (same ladder; `e2/k{k0}/recenter_gm`):

| | start | after the iterations | final hits |
|---|---|---|---|
| $k_0=28$, GM start (4 iterations) | 2.840 | **1.306** | 3/4 (key 0 margin −0.11) |
| $k_0=28$, B start (4 iterations) | 2.233 | 1.244 | 3/4 (key 0 margin −0.23) |
| $k_0=32$, GM start (6 iterations) | 4.129 | **2.739**, still dropping | 1/4 (key 1, +0.17) |
| $k_0=32$, B start (8 iterations*) | 3.652 | 2.579 | 1/4 (key 1, +0.50) |

\*including the four less effective fixed-ε/2 iterations of §4.

- At $k_0=28$ the GM start catches up almost completely: 0.6 nats worse at the start, 0.06 behind after 4 iterations, same hits. At $k_0=32$ it is 0.16 behind and still gaining 0.12–0.24 per iteration.
- So "GM poisons + exact re-centered iteration" gets as far as starting from problem B, without the Alice adjoint and B crafting (about 45 min at $k_0=28$).

**Kept radius over the iterations** — not monotonic:

| run | kept radii |
|---|---|
| $k_0=24$, B start | ε/4, ε/4, ε/4, ε/8 |
| $k_0=28$, B start (4-radius ladder) | ε/4, ε/2, ε/4, ε/8 |
| $k_0=32$, B start (4-radius ladder) | ε/8, ε/4, ε/2, ε/4 |
| $k_0=28$, GM start | ε/2, ε/2, ε/4, ε/2 |
| $k_0=32$, GM start | ε/2, ε/2, ε/8, ε/4, ε/2, ε/8 |

- Consecutive kept radii usually differ by a factor 2, but the GM start at $k_0=32$ dropped twice by a factor 4 (ε/2 → ε/8); the ladder still found it because for r = ε/2 it extends downwards. Neighbouring radii are often within noise of each other.
- The cap at ε/2 is often binding when starting far from the optimum (GM start: 3 of 4 iterations at $k_0=28$, 3 of 6 at $k_0=32$). Larger radii were not tried.

## 7. Finite-difference check of the adjoint gradient ($k_0=32$, real ResNet)

At the iteration-4 poisons (key CE 3.027), the analytic directional derivative of the real objective (adjoint along Bob) is compared with a central finite difference of real Bob trainings (`e2/k32/recenter/fdcheck.json`). Only the 28% of pixels that can move in both directions inside the ε-ball are used.

| ratio real finite difference / adjoint | h = ε/16 | ε/64 | ε/256 |
|---|---|---|---|
| direction −sign(G) (signed step) | 1.11 | 1.19 | 1.17 |
| direction −G (proportional) | 1.15 | 0.88 | 1.27 |

- The adjoint is within about 10–25% of the real derivative, with no large systematic error; the real derivative is, if anything, 11–19% larger in the signed direction (the opposite sign of the 20% shortfall seen in §4), so there is no sign of a gradient bug.
- The real objective is not smooth at this scale: forward and backward differences disagree (proportional direction at ε/256: −1.68 vs −6.21; signed: −41.0 vs −45.9). This fits the ReLU-kink explanation, which is not proven.
- Not covered: the 72% of pixels sitting at the ε boundary, which move only inward in the real steps. So the §4 gap remains unexplained.

## 8. Not done yet

- Diagnostic of the ~20% gap at small radii (running, `keysel/phi_gapdiag.py`): split real/predicted into a dynamics factor and a curvature-of-$J$ factor, locate by epoch where the real and linear displacements diverge, and test the tangent against finite differences of the real SGD map (ReLU-kink signature).
- Transfer test: Bob with another seed and data order on the ladder poisons (postponed).
- Larger trust radii (above ε/2) when starting far from the optimum.
- E3/E4: 10 keys, 3 key sets, GM vs exact iteration, minimal number of images for 2 hits out of 10 at top-1.
