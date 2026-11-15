# Late-window taggants with linearized dynamics — statement of purpose

Follow-up to `docs/phi_identity_results.md`. Theory: T. Kerogues, *Dynamique de l'empoisonnement et Adversarial ML* (the "PDF"); equation numbers below refer to it.

## 1. Purpose

The Φ = I study showed two things: (i) the PDF's linearized transport diverges when it is run over the whole training (R4: $\|e_{\rm lin}\| \approx 10^8$ vs $\|\eta\| = 4.1$), and (ii) a kick-sized perturbation is amplified 33× at epoch 1 but only 1.2× at epoch 20. So the problem is *where* we linearize, not the linearization itself.

As a **methodological step**, we restrict poisoning to a late window $[k_0, N)$ in which the linearized dynamics are measured to be non-expanding, and we model that window exactly as in the PDF (full $\Phi$, no Φ = I shortcut). This lets us:

1. check that $\eta_{\rm lin}$ predicts the real displacement $\eta$ inside the window;
2. compare problem A and problem B on the setting where they are valid;
3. if (1)–(2) hold, compare the dynamic model with plain gradient matching injected over **the same window**, by the minimal number of poisoned images needed to reach the detection p-value threshold;
4. (optional) compare gradient matching over the full training with gradient matching over the late window only.

Relaxing the late-injection assumption (poisons present from the start) is left for later.

## 2. Why the linearization blows up early, and the proxy that detects it

Discrete SGD (no momentum, weight decay $\lambda_{\rm wd}$), with Alice clean and Bob poisoned:
$$\theta^A_{k+1} = \theta^A_k - \gamma_k \nabla L_k(\theta^A_k), \qquad \theta^B_{k+1} = \theta^B_k - \gamma_k \nabla L^\delta_k(\theta^B_k),$$
where $L_k$ includes the $\tfrac{\lambda_{\rm wd}}2\|\theta\|^2$ term. Subtract the two and expand around Alice to get the exact form of the gap $e_k = \theta^B_k - \theta^A_k$:
$$e_{k+1} = M_k e_k - \gamma_k \Delta g_k(\theta^A_k) + r_k, \qquad M_k = I - \gamma_k H_k,\quad H_k = \nabla^2 L_k(\theta^A_k),$$
where $r_k$ collects the second-order term in $e_k$ and the change of the kick between $\theta^A_k$ and $\theta^B_k$. Unrolling it from $e_{k_0} = 0$:
$$e_N = \underbrace{-\sum_{k \ge k_0} \gamma_k\, \Phi_{N,k+1}\, \Delta g_k(\theta^A_k)}_{\eta_{\rm lin}} \;+\; \underbrace{\sum_{k\ge k_0} \Phi_{N,k+1}\, r_k}_{\varepsilon}, \qquad \Phi_{N,k} = M_{N-1}\cdots M_k .$$
The linearization error $\varepsilon$ is the sum of the local residuals **transported by $\Phi$**. Since $H_j$ is symmetric, $\|M_j\|_2 = \max_i |1-\gamma_j\lambda_i(H_j)|$, and
$$\|\Phi_{N,k}\|_2 \;\le\; \prod_{j \ge k} \max_i \big|1 - \gamma_j \lambda_i(H_j)\big| .$$
A factor exceeds 1 exactly when some $\lambda_i(H_j) < 0$ (negative curvature) or $\lambda_i(H_j) > 2/\gamma_j$ (edge of stability, which full-batch and SGD training reach in the high-lr phase [4]). Both hold early, and over $\sim 10^3$ steps even a rate of $1.016$ per step gives $\sim 10^7$, which is what R4 measured. Late in training $\gamma_j$ is small: $\gamma_j\lambda_{\max} < 2$, and the expansion from small negative eigenvalues, $\log \|M_j\| \approx \gamma_j |\lambda^-_{\min,j}|$, adds up to little.

**The bound does not drop below 1, even late.** Each factor is a worst case over all $P \approx 1.1\cdot10^7$ directions, so it is below 1 only if $H_j + \lambda_{\rm wd}I$ is positive definite. Write $H_j = G_j + R_j$:
- the Gauss–Newton term $G_j = \frac1B\sum_i J_i^\top\nabla_f^2\ell\,J_i$ is positive semi-definite but has rank at most $B(n_c-1) \approx 5\cdot10^4$ ($n_c = 100$ classes), so it vanishes on more than 99% of directions;
- on those directions only $R_j = \frac1B\sum_i\sum_c \partial_{f_c}\ell\,\nabla^2_\theta f_c$ is left, and it is indefinite.

Published Hessian spectra show exactly this: a few large outliers, a huge bulk near 0, and small negative eigenvalues that persist until the end of training [1, 2, 3]. One eigenvalue below $-\lambda_{\rm wd} = -5\cdot10^{-4}$ is enough to make a factor $1+\gamma_j(\mu-\lambda_{\rm wd}) > 1$, so the product stays $\ge 1$. Late in training it grows only slowly, for example $\gamma = 0.005$, $\mu = 0.1$ over 1000 steps gives $e^{0.5}\approx 1.6$.

The quantity of interest is the **finite-time growth of the transport**, $\|\Phi_{N,k}\|_2$. It is the finite-time analogue of the top Lyapunov exponent, $\Lambda_k = \frac{1}{N-k}\log\|\Phi_{N,k}\|_2$. The product bound only *explains* the early blow-up; we do not use it as the criterion. The true norm is smaller than the bound, because the negative eigenvectors change from step to step. Still, on directions where all the $H_j$ are near zero, $\Phi$ only shrinks by $\prod_j(1-\gamma_j\lambda_{\rm wd}) \approx 1$, so even $\|\Phi_{N,k}\|_2 \approx 1$ at best. The linearization needs **no amplification**, not contraction, so the criterion is $\|\Phi\| \le C$ with $C$ a small constant, not $\|\Phi\| < 1$.

**Criterion for $k_0$.** Since Alice's full trajectory is known (white-box), we measure $\|\Phi_{N,k}\|_2$ directly by power iteration on $\Phi_{N,k}^\top\Phi_{N,k}$. Each iteration is one forward tangent pass ($v \mapsto M_j v$) and one adjoint pass ($p \mapsto M_j^\top p$) over $[k, N)$, both made of Hessian-vector products. We do this for $k$ on a grid of epoch boundaries. The injection start is then
$$k_0 = \min\Big\{k \;:\; \max_{k' \ge k} \|\Phi_{N,k'}\|_2 \le C\Big\}.$$
The max over $k'$ is needed because every poison step $k' \ge k_0$ is transported by its own $\Phi_{N,k'+1}$.

**Choosing $C$: stability vs poisoning power.** The two sides of the trade-off are explicit:
- *Stability:* $\|\varepsilon\| \le C \sum_k \|r_k\|$, so the relative error on $\eta$ is about $C$ times the relative size of the local residuals. These residuals do not vanish (ReLU kinks, kick evaluated at $\theta^A$ instead of $\theta^B$).
- *Power:* the displacement scales with the learning-rate mass left in the window, $\sum_{k\ge k_0}\gamma_k$. Under the cosine schedule the remaining fraction after epoch $e$ is $R(t) = (1-t) - \sin(\pi t)/\pi$, $t = e/E$:

| $k_0$ (epoch) | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| lr mass left | 44% | 30% | 18% | 10% | 4% | 1.3% |

$C$ is therefore not fixed a priori but calibrated. From the curve $k \mapsto \|\Phi_{N,k}\|$ we take 3–4 candidate values of $k_0$ where it flattens (e.g. epochs 16, 20, 24, 28). At each we run the validation below, and we keep the **earliest** $k_0$ that passes. $C := \max_{k\ge k_0}\|\Phi_{N,k}\|$ is then an output. Calibration uses the first key set and is frozen for all repetitions.

A bounded $\Phi$ is necessary but not sufficient: the residuals $r_k$ must also stay small. Before comparing A and B we therefore check, inside the window, that $\eta_{\rm lin}$ matches the real $\eta$: $\cos(\eta_{\rm lin}, \eta)$, $\|\eta_{\rm lin}\|/\|\eta\|$, and $q^\top\eta_{\rm lin}$ vs $q^\top\eta$ vs $\Delta J$. This is R4 restarted at $k_0$. Default pass thresholds: $\cos(\eta_{\rm lin},\eta) \ge 0.9$, $\|\eta_{\rm lin}\|/\|\eta\| \in [0.8, 1.25]$, $q^\top\eta_{\rm lin}$ within 25% of $q^\top\eta$.

## 3. Formulas used from the PDF (restricted to the window)

All sums and recursions run over $k \in [k_0, N)$ with $e^{\rm lin}_{k_0} = 0$. The adjoint only needs to be propagated back to $k_0$. Poisons sit in their own batch, so $\Delta g_k = 0$ except at poison steps.

**Kick** (66):
$$\Delta g_k(\theta;\delta) = \nabla L^\delta_k(\theta) - \nabla L_k(\theta) = \frac{1}{|B_k|}\sum_{i\in B_k\cap\mathcal P}\big[g^\delta_i(\theta) - g_i(\theta)\big],\qquad g_i^\delta(\theta) = \nabla_\theta \ell(\theta; x_i+\delta_i, y_i).$$

**Linearized forward dynamics** (67)–(68):
$$e^{\rm lin}_{k+1} = M_k e^{\rm lin}_k - \gamma_k \Delta g_k(\theta^A_k;\delta), \qquad \eta_{\rm lin}(\delta) = e^{\rm lin}_N, \qquad M_k = I - \gamma_k H_k .$$

**Exact and approximate problems** (17), (18), (20), with $J$ the mean CE on the keys toward their target classes:
$$\text{exact: } \min_{\delta\in\mathcal C} J(\theta^B_N(\delta)), \qquad \text{(A): } \min_{\delta\in\mathcal C} J\big(\theta^A_N + \eta_{\rm lin}(\delta)\big), \qquad \text{(B): } \min_{\delta\in\mathcal C} \nabla J(\theta^A_N)^\top \eta_{\rm lin}(\delta).$$

**Admissible set** (44):
$$\mathcal C = \prod_{i\in\mathcal P}\mathcal C_i,\qquad \mathcal C_i = \{\delta_i : \|\delta_i\|_\infty \le \epsilon,\ x_i + \delta_i \in [0,1]^d\}.$$

**Discrete adjoint of B** (69)–(70):
$$p_N = q = \nabla J(\theta^A_N), \qquad p_k = M_k^\top p_{k+1}, \qquad q^\top e^{\rm lin}_N = -\sum_{k} \gamma_k\, p_{k+1}^\top \Delta g_k(\theta^A_k;\delta).$$

**Separable subproblems of B** (42), (71), (73), where $\mathcal K_i$ is the set of steps at which image $i$ is presented:
$$\min_{\delta\in\mathcal C}\sum_i F_i(\delta_i) = \sum_i \min_{\delta_i\in\mathcal C_i} F_i(\delta_i), \qquad F_i(\delta_i) = -\sum_{k\in\mathcal K_i}\frac{\gamma_k}{|B_k|}\, p_{k+1}^\top\big[g^\delta_i(\theta^A_k) - g_i(\theta^A_k)\big],$$
$$\nabla_{\delta_i} F_i = -\sum_{k\in\mathcal K_i}\frac{\gamma_k}{|B_k|}\big[D_{\delta_i} g^\delta_i(\theta^A_k)\big]^\top p_{k+1} \quad (\text{autodiff of } p_{k+1}^\top g^\delta_i).$$

**Projected gradient step** (47):
$$\delta^{(t+1)}_i = \Pi_{\mathcal C_i}\big(\delta^{(t)}_i - \alpha_t \nabla F_i(\delta^{(t)}_i)\big).$$

**Problem A** (75)–(77): forward pass to $\widehat\theta_B = \theta^A_N + e^{\rm lin}_N$, then the adjoint with a δ-dependent terminal condition:
$$\mathcal J_A(\delta) = J(\widehat\theta_B), \qquad p_N = \nabla J(\widehat\theta_B),\quad p_k = M_k^\top p_{k+1}, \qquad \nabla_{\delta_i}\mathcal J_A = -\sum_{k\in\mathcal K_i}\frac{\gamma_k}{|B_k|}\big[D_{\delta_i} g^\delta_i(\theta^A_k)\big]^\top p_{k+1}.$$
Each A iteration costs one forward and one adjoint propagation over the window. B's adjoint is computed once.

**Combined strategy** (§2.4): solve B, set $\delta^{(0)}_A = \delta_B$, run a few projected-gradient iterations on A (63), and keep the best iterate according to $\mathcal J_A$. Then evaluate by actually training Bob.

## 4. Experiments

Setup as in `docs/phi_identity_results.md` §2 (CIFAR-100, ResNet18-GN, plain SGD, 40 epochs, poisons in their own last batch each epoch), except that poisons are present only in epochs $\ge k_0$. Bob always uses **Alice's seed and data order (s0)** and starts from Alice's $\theta^A_{k_0}$, so $e_{k_0} = 0$ exactly and the only modelling error is the linearization.

**Keys and repetitions.** 10 keys, as in the paper. Within a repetition, every method uses the same 10 keys. The experiment is repeated with **3 independent key sets** (same Alice) to average out the effect of noise sinks.

**Methods** (all injected on $[k_0, N)$, same images to poison, same $\epsilon$):

| Method | Crafting | Restarts |
|---|---|---|
| Gradient matching (GM) | signed-Adam gradient matching on the final Alice model $\theta^A_N$ (`keysel.craft`) | 8, keep best |
| Problem B | separable subproblems with the adjoint over $[k_0, N)$ | 8, keep best by the B objective |
| B → A | up to 5 projected-gradient iterations of A starting from the best $\delta_B$, keep best by $\mathcal J_A$ | — |

| Step | What | Output |
|---|---|---|
| E1 | $\|\Phi_{N,k}\|_2$ on epoch boundaries (power iteration, forward + adjoint HVPs); product bound at a few epochs to illustrate §2 | curve $k\mapsto\|\Phi_{N,k}\|$, candidate $k_0$ |
| E2 | Validation at each candidate $k_0$: craft with B, train Bob from $k_0$, compare $\eta_{\rm lin}$ and $\eta$ | chosen $k_0$ and $C$; go/no-go for E3 |
| E3 | GM vs B vs B→A: hits, margins, key CE, test accuracy, p-value; 3 key sets | comparison table |
| E3b | For B→A, objective $\mathcal J_A$ and real key CE/margins of Bob after each of the 5 A iterations (one plot, run once) | is A worth its cost? |
| E4 | Minimal number of poisoned images per key reaching the detection threshold, for GM and the best dynamic method | budget curves |
| E5 (optional) | GM over the full training vs GM over $[k_0,N)$ | same metrics as E3 |

Detection uses the **top-1** binomial test of `detect.py` (`--top-k 1`): with $K = 10$ keys, $p = \Pr[\mathrm{Bin}(10, 1/100) \ge \text{hits}]$. One hit gives $p = 0.096$, two hits give $p = 4.3\cdot10^{-3}$, so the detection threshold ($\alpha = 0.05$ or $0.01$) is reached at **2 hits out of 10**.

## 5. Open points

- Pass thresholds of the E2 validation (defaults in §2).
- The sink class (78 for this recipe, §3.6 of the Φ = I results) makes the binomial null miscalibrated for noise keys; the 3 key sets mitigate but do not remove this.

## References

1. L. Sagun, L. Bottou, Y. LeCun. *Eigenvalues of the Hessian in Deep Learning: Singularity and Beyond.* arXiv:1611.07476, 2016.
2. B. Ghorbani, S. Krishnan, Y. Xiao. *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML 2019.
3. V. Papyan. *Measurements of Three-Level Hierarchical Structure in the Outliers in the Spectrum of Deepnet Hessians.* ICML 2019.
4. J. Cohen, S. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021.
