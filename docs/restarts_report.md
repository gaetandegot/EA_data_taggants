# Restarts barely change the optimized score

**Summary.** The best cosine over restarts, $S_j$, is almost entirely a property of the key; the random initialization of the perturbations barely affects it:
- on CIFAR-100, the spread across restarts of one key is 5–7× smaller than the spread across keys;
- 95–98% of the variance of a single restart's score is due to the key;
- choosing the 10 kept keys out of 20 with 1 restart instead of 8 changes at most one key in 97% of draws.

## Setup

- **Model:** CIFAR-100. Alice ($\theta_A$) is a ResNet-18 trained without augmentation (61% test accuracy).
- **Signing set:** 50 training images of class $y_j$.
- **Crafting:** gradient matching on $\delta$ ($\|\delta\|_\infty\le16/255$), signed Adam (lr 0.1, MultiStepLR) for 240 steps, model in eval mode, no augmentation, no perceptual term.
- **Restarts:** each restart draws a new initialization $\delta_0\sim\mathcal{N}(0,\varepsilon)$, clipped to the $L_\infty$ ball and the pixel range. Without augmentation, this initialization is the only source of randomness between restarts.
- **Recorded value:** the cosine $c_{j,r}$ between the key gradient and the summed signing gradient at the end of restart $r$, without augmentation. $S_j=\max_r c_{j,r}$.

| Set | Keys | Restarts per key | Cosine |
|---|---|---|---|
| C1 | 100 random keys | 8 | full parameter vector (the paper's objective) |
| E2 | 40 keys with the sink label $y_j=39$ | 4 | full; also the last-layer cosine at the same restarts |

## Measures

- **Within-key SD / range:** standard deviation and range (max − min) of $c_{j,\cdot}$ over the restarts of one key, averaged (and maximized) over keys.
- **Between-key SD / range:** standard deviation and range of $S_j$ across keys.
- **Gain of the best restart:** $S_j-c_{j,1}$, the improvement of the best restart over a single one.
- **ICC(1):** one-way intraclass correlation, the share of the variance of a single restart's score that is due to the key.
- **Split-half reliability:** Spearman correlation between the best of the first half of the restarts and the best of the second half.
- **Single vs best:** Spearman correlation between $c_{j,1}$ and $S_j$.

## Results

| | C1, full (8 restarts) | E2, full (4 restarts) | E2, last layer (4 restarts) |
|---|---|---|---|
| Within-key SD, mean (max) | 0.0016 (0.0056) | 0.0013 (0.0026) | 0.0011 (0.0031) |
| Within-key range, mean (max) | 0.0047 (0.0178) | 0.0029 (0.0055) | 0.0024 (0.0074) |
| Between-key SD of $S_j$ | 0.0108 | 0.0061 | 0.0029 |
| Between-key range of $S_j$ | 0.053 | 0.029 | 0.012 |
| Gain of best restart, mean (max) | 0.0023 (0.0091) | 0.0014 (0.0055) | 0.0010 (0.0043) |
| ICC(1) | **0.98** | **0.95** | 0.85 |
| Split-half reliability | **0.99** | **0.92** | 0.95 |
| Spearman single vs best | 0.99 | 0.93 | 0.95 |

For the last-layer column, the "best" restart is the maximum of the last-layer cosine. In the protocol, the restart is chosen by the full cosine.

**Effect on selection (C1).** Over 20,000 draws of 20 keys, we compare the 10 keys kept by ranking on $c_{j,1}$ (one restart) with those kept by ranking on $S_j$ (8 restarts):
- on average, the two sets share 9.5 of their 10 keys;
- they are identical in 54% of draws;
- they differ by exactly one key in 43% of draws and by two in 3%;
- three differences occur in 0.03% of draws.

## Interpretation

- **The key, not the restart, determines the score.** The typical within-key spread (SD about 0.0015) is small compared with the differences between keys (SD 0.006–0.011). The best restart adds only about 0.002 to a single restart. Ranking keys by $S_j$ is therefore ranking keys by a stable property of $(x_j,y_j)$ and its signing set.
- **The draft's restarts argument does not hold here.** The draft argues that taking the best of 8 restarts strengthens the selection effect by removing optimization noise (the "optimizer's curse"). In this setting there is almost no such noise to remove, so restarts neither create nor amplify the bias. The bias comes from what the score measures (Section 5.3), not from how many restarts are run.

## Limits

- **CIFAR only.** On ImageNet, I3 recorded only the best of its 4 restarts, and the E5 pilot used 1 restart, so the within-key variance at ImageNet scale was not measured.
- **No augmentation and no perceptual term.** The paper crafts with differentiable augmentation and an LPIPS penalty. Both add stochasticity across restarts, so the restart-to-restart variance in the paper's own protocol may be larger than measured here.
- **Reproduce:** `python -m keysel.restarts` (numbers in `docs/figures/restarts.json`).
