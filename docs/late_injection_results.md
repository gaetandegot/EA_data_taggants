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
8. The ~20% gap between predicted and real gain at small steps is not a gradient bug (§7, §8). It comes from Bob's training dynamics, not from $J$, and the cause is that the real SGD map of this ReLU network is not smooth: over one epoch, any perturbation produces a response of roughly fixed size (about 2e-3 in parameter norm), on top of the linear one (§8).
9. **With a smooth activation (SiLU) the gap closes at small steps and the ladder converges much faster** (§9): real/predicted is 0.93 at ε/32 and 0.98 at ε/128 (ReLU: 0.79 and 0.84), the one-epoch floor of the SGD map drops by at least 140×, and the ladder from B's poisons reaches **4 of 4 hits at iteration 2** (key CE 0.73) and CE 0.37 after 4 iterations at $k_0=32$ (ReLU: 2.58, 1 of 4). At the radii the ladder actually uses (ε/8 to ε/2) real/predicted is still only 0.2–0.6, so the model stays a direction finder, not a predictor.
10. On SiLU the anchored B is **not** better than GM (CE 2.77 / 1 hit vs 2.34 / 2 hits at $k_0=32$), the reverse of ReLU. The cause is untested; B-cos (§9) is the test, with results pending.

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
- **Gap at small radii:** real/predicted stalls at about 0.8 between ε/32 and ε/512 although the anchoring is perfect. §8 shows why: the real training map of this ReLU network is not smooth (a fixed-size response of about 2e-3 per epoch from ReLUs switching), so no first-order model can be accurate at very small steps. It is not a gradient bug (§7).
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

## 8. Diagnostic of the gap between predicted and real gain ($k_0=32$, iteration-4 poisons)

`keysel/phi_gapdiag.py` → `e2/k32/recenter/gapdiag.json`; even-part test → `even_part.json`.

**Split of real/predicted.** For a candidate step, predicted = linear model; the real step is checked by training Bob.

| radius | real / predicted | = dynamics factor | × curvature of $J$ | cos(real, linear displacement) | ‖real‖ / ‖linear‖ |
|---|---|---|---|---|---|
| ε/32 | 0.788 | 0.840 | 0.938 | 0.66 | 1.35 |
| ε/128 | 0.842 | 0.859 | 0.980 | 0.40 | 2.22 |

- The adjoint along Bob and the forward tangent give the same prediction to 4e-5 on the real network: no inconsistency in the code.
- **$J$ behaves like a smooth function** (its shortfall drops from 6% to 2% for a 4× smaller step). **The floor is in the dynamics** (about 0.85 at both radii).
- The real and linear displacements are identical right after the first poison step (by construction) and already differ after the 97 clean steps of the next epoch (cos 0.75, real 1.33× larger at ε/32); later epochs carry the difference without amplifying it. Most of the extra displacement is invisible to the keys ($q^\top$ is still 84–86% right while the full vectors disagree by ~100%).

**Even part of the real SGD map.** $E(s)=\|\tfrac12[S(\theta+sv)+S(\theta-sv)]-S(\theta)\|$ for a random direction $v$ of norm 0.016, at epoch 33. The slope of $\log E$ vs $\log s$ is 2 for a smooth map, about ½ for randomly crossed gradient jumps (ReLU kinks), and 0 for a fixed floor.

| perturbation norm | 1.6e-2 | 4.0e-3 | 1.0e-3 | 2.5e-4 | 6.3e-5 | 1.6e-5 |
|---|---|---|---|---|---|---|
| single poison step: $E$ | 3.0e-4 | 1.4e-4 | 6.8e-5 | 3.6e-5 | 3.1e-5 | 2.7e-5 |
| slope | | 0.53 | 0.54 | 0.46 | 0.11 | 0.09 |
| one epoch (98 steps): $E$ | 2.06e-3 | 1.93e-3 | 1.90e-3 | 1.85e-3 | 1.83e-3 | 1.83e-3 |
| slope | | 0.05 | 0.01 | 0.02 | 0.01 | 0.00 |

- **One step:** randomly crossed ReLU kinks (slope ≈ ½) down to a perturbation of about 2.5e-4, then a floor of about 3e-5, the size expected from float32 rounding.
- **One epoch:** a floor of about 2e-3 already from a perturbation of 4e-3 down to 1.6e-5 (three decades). At the smallest sizes the odd part is flat too (about 1.1e-3), so the whole response is the floor. Over 97 steps × 512 images, any perturbation, however small, flips some ReLUs, and each flip changes an update by a fixed amount that is carried forward. So the training map behaves as if it were discontinuous at a scale of about 2e-3.
- This explains the gap: at ε/32 the real displacement (norm ≈ 0.016) carries an extra response of about 2e-3 (≈ 13%), at ε/128 a much larger share, which is why the real displacement grows relative to the linear one and turns away from it as the step shrinks. The kink test of the tangent against finite differences shows the same thing (relative error 0.32, 0.68, 1.71 at scales 1, ¼, 1/16 on one epoch).
- Consequence for the method: no first-order model can be accurate below a step of roughly this size; the exact gradient remains useful for larger steps (ε/16 to ε/4), where the real gain is 40–80% of the prediction.

**Direct count of ReLU switches** (`keysel/phi_switches.py` → `e2/k32/recenter/switches.json`; same state and random direction as the even-part test). A switch is an (image, unit) pair whose ReLU sign differs between the two parameter vectors.

At one state ($\theta$ vs $\theta+hv$):

| perturbation norm | 1.6e-2 | 4.0e-3 | 1.0e-3 | 2.5e-4 | 6.3e-5 | 1.6e-5 |
|---|---|---|---|---|---|---|
| switches, poison batch (400 images) | 17,967 | 4,564 | 1,106 | 281 | 91 | 56 |
| switches, clean batch (512 images) | 21,961 | 5,515 | 1,308 | 344 | 103 | 39 |
| log-log slope (poison / clean) | | 0.99 / 1.00 | 1.02 / 1.04 | 0.99 / 0.96 | 0.81 / 0.87 | 0.35 / 0.70 |

Along one epoch (two trajectories started at $\theta$ and $\theta+hv$, switches counted at every step on that step's batch):

| initial perturbation | switches at step 0 | mean, steps 0–9 | mean, last 10 steps | distance after the epoch |
|---|---|---|---|---|
| 1.6e-2 | 17,967 | 34,560 | 74,630 | 1.6e-2 (1.0×) |
| 1.0e-3 | 1,106 | 10,997 | 61,804 | 2.4e-3 (2.4×) |
| 6.3e-5 | 91 | 7,520 | 60,376 | 2.1e-3 (33×) |
| 1.6e-5 | 56 | 10,238 | 64,805 | 2.1e-3 (137×) |

- **One step:** the number of switches is proportional to the perturbation (slope 1) over almost three decades. Even a perturbation of norm 2.5e-4 switches about 300 activations per batch, so the autodiff linearization of a single step is never exact at practical step sizes. Below about 1e-4 the slope flattens, where the even-part test also reached its float32 floor.
- **One epoch (cascade):** the late switch count (about 60,000–75,000 per step) barely depends on the starting size, over a factor 1000, and every small perturbation grows to a distance of about 2e-3, the floor measured by the even-part test. This directly confirms the cascade explanation of §8.

## 9. Smooth activation: ResNet18-GN with SiLU ($k_0=32$)

Purpose: remove the ReLU kinks that §8 blames for the predicted-vs-real gap. Run `/Data/gaetan.degot/checkpoints/phi_r18gn_silu_s0`: the same run (same data, seed, keys, 40 epochs, lr 0.05, weight decay 5e-4, cosine schedule, batch 512, ε = 16/255; the config was not retuned) with `resnet18_gn_silu` (every ReLU replaced by SiLU; `keysel/run_phi_arch.py`). Alice's test accuracy is 0.448; her key CE is 7.17 (margins −6.6, −6.5, −6.7, −5.1), different from the ReLU Alice (6.59), so numbers below are not comparable one-to-one with §2–§8. Same caveats: 4 keys, one seed.

### Anchored B vs GM vs random (E2, `e2/k32/results.json`)

| poisons | hits | key margins | key CE (Alice 7.17) | real $q^\top\eta$ | real $\Delta J$ | $\|\eta\|$ | mean kick anchoring cos (epochs 33–39) |
|---|---|---|---|---|---|---|---|
| problem B | 1/4 (key 2) | −2.8, −0.4, +0.5, −1.7 | 2.77 | −70.9 | −4.39 | 0.76 | 0.03 |
| gradient matching | 2/4 (keys 1, 2) | −3.7, +1.3, +1.7, −0.7 | **2.34** | −7.9 | −4.83 | 0.12 | 0.55 |
| random noise | 0/4 | −6.9, −5.9, −6.0, −4.9 | 6.90 | −0.27 | −0.27 | 0.034 | 0.99 |

- **GM beats anchored B here (2.34 vs 2.77, 2 vs 1 hits), the opposite of ReLU** (§2: B 3.65, GM 4.13). The cause is unknown. Observations: B moves the parameters 6× farther (0.76 vs 0.12) and its kicks at Bob's parameters are almost unrelated to those at Alice's (cos 0.03, norm ratio 0.06–0.08); the real $q^\top\eta$ (−70.9) is even larger than the predicted −61.2 (ratio 1.16, $\cos(\eta_{\rm lin},\eta)=0.66$), yet $\Delta J$ is only −4.39, because $J$ saturates when the margins grow. Hypothesis (untested): B overshoots the linear objective, so it spends its budget on a direction that the linear model rates highly but that does not translate into key CE. One seed of keys, so it could also be luck.
- Reading the large $q^\top\eta$ as agreement between model and reality would be wrong: it is not a validation, only $\Delta J$ and the hits count.

### The gap at small steps (`recenter/gapdiag.json`, B poisons, SiLU)

| radius | predicted $\Delta J$ | real $\Delta J$ | real / predicted | = dynamics factor | × curvature of $J$ | cos(real, linear displacement) |
|---|---|---|---|---|---|---|
| ε/32 | −0.230 | −0.214 | **0.930** (ReLU 0.788) | 0.946 (0.840) | 0.983 (0.938) | 0.9998 (0.66) |
| ε/128 | −0.054 | −0.053 | **0.984** (ReLU 0.842) | 0.989 (0.859) | 0.995 (0.980) | 1.0000 (0.40) |

(ReLU values in brackets, from §8, taken at the iteration-4 poisons, so not the same poisons.) The gap closes as the step shrinks, and the real and linear displacements agree in direction and norm (norm ratio 0.998). The kink test (tangent vs finite differences of the real SGD map) gives ratio 1.000 and relative error 2e-4 to 1e-3 at all three scales, against 0.32–1.71 on ReLU. This confirms that the 20% gap of §8 comes from the ReLU kinks.

**Even part of the real SGD map** (`even_part.json`, B poisons, epoch 33, perturbation norm 0.016 down to 1.6e-5; "one epoch" = the 98-step span):

| | one poison step | one epoch |
|---|---|---|
| SiLU, $E$ at norm 1.6e-2 | 2.5e-6 | 1.4e-5 |
| SiLU, $E$ at norm 1.6e-5 | 5.1e-7 | 3.0e-6 |
| ReLU (§8), $E$ at 1.6e-2 / 1.6e-5 | 3.0e-4 / 2.7e-5 | 2.1e-3 / 1.8e-3 |

The nonlinear response is 120× (step) to 140–600× (epoch) smaller than on ReLU, and the measured log-slope is 0.1–0.4, not 2: at these sizes the response sits near the float32 rounding floor, so the expected slope 2 of a smooth map is not resolved. What the test shows is the absence of the fixed 2e-3 response, not a measured slope of 2.

### Ladder from B's poisons (`e2/k32/recenter`, 3-radius ladder, adaptive)

| iteration | kept radius | predicted $\Delta J$ → real | key CE (Alice 7.17) | hits (margins) |
|---|---|---|---|---|
| start (B) | | | 2.774 | 1/4 |
| 1 | ε/4 | −3.39 → −1.28 | 1.492 | 2/4 |
| 2 | ε/2 | −3.69 → −0.76 | 0.732 | **4/4** (0.86, 3.49, 1.70, 0.73) |
| 3 | ε/8 | −0.75 → −0.23 | 0.507 | 4/4 |
| 4 | ε/16 | −0.25 → −0.13 | **0.373** | 4/4 (1.87, 3.77, 3.05, 1.39) |

- Compared with ReLU at the same $k_0$ (CE 2.58, 1/4 hits after 4 ladder iterations plus 4 fixed-radius ones), the SiLU run reaches 4/4 hits after two iterations. Part of this is the smoother problem, part may be the different Alice and keys; not separated.
- **Real/predicted at the radii the ladder actually uses is still low.** For the candidates tried: ε/2 0.17, 0.21 (iterations 1, 2) and the real gain turns positive at iterations 3 (+0.42); ε/4 0.38, 0.31; ε/8 0.62, 0.56, 0.30; ε/16 0.53. So the smooth activation removes the floor at ε/32 and below, but the first-order model is still only a direction finder at ε/8 and above. Likely culprits are the curvature of $J$ and the nonlinear transport, but this was not measured at these radii.
- Only 4 of the 8 planned iterations were done: all user processes were killed at about 06:35 during iteration 5 (cause unverified). The ladder resumes with the same command (`--init B`). The ladder CE is still falling by 0.13 per iteration at iteration 4.

### Restarts: 2 vs 8 (SiLU run, gradient matching) and B-cos at $k_0=32$, $28$

The Data Taggants paper uses 8 restarts for gradient matching; our baseline used 2 (also for B and B-cos). GM was re-crafted with 8 restarts on Alice's final model (`gm8/poisons.pt`; same images, keys and random seeds, so the first two restarts are the earlier ones).

| | key 0 | key 1 | key 2 | key 3 |
|---|---|---|---|---|
| GM match score (cosine, full / head), 2 restarts | 0.725 / 0.883 | 0.896 / 0.948 | 0.932 / 0.947 | 0.860 / 0.907 |
| 8 restarts | 0.728 / 0.887 | 0.897 / 0.947 | 0.932 / 0.944 | 0.865 / 0.906 |
| real Bob margins at $k_0=32$, 2 restarts | −3.73 | +1.29 | +1.68 | −0.74 |
| 8 restarts | −3.75 | +1.25 | +1.61 | −0.80 |

- Real key CE at $k_0=32$: 2.335 with 2 restarts, 2.376 with 8 (2/4 hits both, same keys). The extra restarts changed the match score by at most 0.005 and the real outcome not at all within noise.
- Restarts hardly differ for B-cos either: the weighted-cosine objective of restart 0 vs 1 differs by 0.1–2% (for example group scope at $k_0=32$: −0.1745 / −0.1760, −0.3863 / −0.3934, −0.4461 / −0.4474, −0.3942 / −0.3887).
- So, in this setting, restarts add little: the objectives behave as if they had one dominant basin, and the best of several random starts is not much better than the first. One seed, 4 keys, SiLU only; the ReLU run and other seeds were not tested, so this should be rechecked there before stating it as a general result. The 2-restart comparisons above (GM vs B vs B-cos) are not weakened by it.

### B-cos (cosine version of problem B, `keysel/phi_bcos.py`) at $k_0=32, 28$

| $k_0$ | poisons | hits | key CE | real $\Delta J$ | $\|\eta\|$ | kick anchoring (mean cos) |
|---|---|---|---|---|---|---|
| 32 | B | 1/4 | 2.77 | −4.39 | 0.76 | 0.03 |
| 32 | GM | 2/4 | **2.34** | −4.83 | 0.12 | 0.55 |
| 32 | B-cos, group scope | 0/4 | 3.89 | −3.28 | 0.10 | 0.43 |
| 32 | B-cos, global scope | 1/4 | 2.98 | −4.19 | 0.095 | 0.71 |
| 28 | B | 2/4 | 1.98 | −5.19 | 1.22 | |
| 28 | GM | 3/4 | **1.48** | −5.68 | 0.27 | |
| 28 | B-cos, group scope | 2/4 | 2.59 | −4.58 | 0.26 | 0.39 |

(Alice's key CE 7.17. The $k_0=28$ B and GM anchoring values were not re-read; $k_0=28$ global scope and $k_0=24$ were not finished.)

- B-cos does not beat GM and, at $k_0=32$, not B either. It removes B's overshoot (‖η‖ 0.10 vs 0.76), but the key CE is still worse.
- Scope matters at $k_0=32$ (global 2.98 vs group 3.89), consistent with the caveat that the adjoint is for the mean over keys, so the group scope aligns each key with the joint target; one $k_0$ only.
- GM is the best of the anchored methods on SiLU at $k_0=32$ and $28$. Linear score of the B objective at $k_0=32$: B −61.2, GM −15.3, B-cos −12.5 (group) / −9.2 (global): B wins its own objective by far and loses on the real one.

### Ladder from the 8-restart GM poisons (SiLU, `e2/k{k0}/recenter_gm8`)

Start: GM poisons with 8 restarts (see above), same 3-radius adaptive ladder (ε/2, ε/4, ε/8 first, then {r/2, r, 2r}), 120 steps, 1 restart; the kept candidate is the lowest real key CE. Alice's key CE is 7.17. "Kick anchoring" is the mean cosine between the kicks at Bob's and at the previous trajectory's parameters.

| $k_0$ | iteration | kept radius | predicted $\Delta J$ → real | real / predicted | key CE | hits (margins) | kick anchoring |
|---|---|---|---|---|---|---|---|
| 32 | start | | | | 2.376 | 2/4 (−3.75, 1.25, 1.61, −0.80) | |
| | 1 | ε/4 | −1.76 → −0.86 | 0.49 | 1.513 | 3/4 | 0.89 |
| | 2 | ε/2 | −2.31 → −0.59 | 0.26 | 0.925 | 4/4 (0.19, 1.73, 2.82, 0.07) | 0.85 |
| | 3 | ε/8 | −0.85 → −0.36 | 0.43 | 0.560 | 4/4 | 0.97 |
| | 4 | ε/8 | −0.29 → −0.09 | 0.30 | **0.474** | 4/4 (1.93, 2.87, 3.34, 0.64) | 0.96 |
| 28 | start | | | | 1.522 | 3/4 (−2.77, 3.75, 2.90, 1.25) | |
| | 1 | ε/2 | −4.83 → −0.97 | 0.20 | 0.557 | 4/4 | 0.73 |
| | 2 | ε/4 | −2.49 → −0.30 | 0.12 | **0.254** | 4/4 (2.00, 4.58, 3.30, 1.73) | 0.79 |
| 24 | start | | | | 1.111 | 3/4 (−1.57, 4.95, 3.87, 2.38) | |
| | 1 | ε/4 | −2.37 → −0.89 | 0.37 | 0.225 | 4/4 | 0.89 |
| | 2 | ε/2 | −1.46 → −0.15 | 0.10 | **0.079** | 4/4 (4.38, 5.40, 4.34, 3.02) | 0.76 |

- **Four of four hits after 1–2 iterations at every start epoch**, from a GM start with 2/4 (k0=32) or 3/4 hits (k0=28, 24). Key CE after the iterations: 0.47, 0.25, 0.08 at k0=32, 28, 24: lower for earlier injection, as before.
- At $k_0=32$ the earlier ladder from B's poisons reached 0.373 after 4 iterations (above); from GM it is 0.474, so the GM start is slightly behind after the same number of iterations. Different starting poisons and one seed, so I do not read this as a significant difference.
- **Kept radii** are in 0.125–0.5 ε (ε/8 to ε/2) and not monotonic (k0=32: ε/4, ε/2, ε/8, ε/8). Real/predicted at the kept radius is 0.1–0.5, lower at the larger radii, as at the ladder radii of the B-start runs.
- The stop rule (kept gain < 0.02) has not fired at any $k_0$. At $k_0=32$ the last real gain is −0.086 (still moving).
- Not finished: k0=28 and 24 have two iterations each. The plan is two more for both (4 in total); the k0=32 ladder is at four, so it is not continued. A queued run was killed on 2026-11-17 at about 06:30 (the iteration in progress at k0=28 was lost).

### Pending (queued for the GPU window)

- Two more ladder iterations from the 8-restart GM poisons at $k_0=28$ and $24$ (4 iterations each in total; $k_0=32$ stays at 4); queued for the 2026-11-17 evening window with `KS="28 24" ROUNDS=1`.
- B-cos at $k_0=28$ with global scope and at $k_0=24$ (needs E2 there first): not run; low priority after the results above.

## 10. Not done yet

- Transfer test: Bob with another seed and data order on the ladder poisons (postponed).
- Larger trust radii (above ε/2) when starting far from the optimum.
- E3/E4: 10 keys, 3 key sets, GM vs exact iteration, minimal number of images for 2 hits out of 10 at top-1.
