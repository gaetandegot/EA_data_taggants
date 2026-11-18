# Related work for the late-window, white-box taggant study

Written 2026-11-16 from web searches and partial reads, to check what is already known before claiming anything. **Verification levels** used below:

- **Read**: I read the passage itself (verbatim text of the paper or of its HTML rendering).
- **Summary**: I only have a model-extracted summary of the paper body, or a search snippet. Check the paper before citing.
- **Repo**: verified in this repository's code.

This is not a literature review: about ten searches and four fetches. Absence of a result below means "not found", not "does not exist".

Our setting, for reference: CIFAR-100, ResNet18-GN, plain SGD, Bob equals Alice (same init, same data order) until epoch $k_0$, poisons replace clean images in the poison batch of every epoch from $k_0$ on; the owner knows the whole training procedure (white box); poisons are crafted with the exact first-order gradient of the real objective (adjoint along Bob's trajectory); results in `docs/late_injection_results.md`.

## 1. Differentiating through training to craft poisons

| Work | What it does | Verification |
|---|---|---|
| Muñoz-González et al. 2017, [arXiv 1708.08689](https://arxiv.org/abs/1708.08689), "Towards poisoning of deep learning algorithms with back-gradient optimization" | Computes the gradient of the attacker's objective through the training run by automatic differentiation, reversing the learning procedure to avoid storing the trajectory. | Abstract: Read. Body: Summary. |
| Huang et al. 2020, [arXiv 2004.00225](https://arxiv.org/pdf/2004.00225), "MetaPoison" (NeurIPS 2020) | Clean-label poisons by unrolling $K=2$ SGD steps, over an ensemble of surrogate models staggered by epoch. | Sections 1–4 and Appendices A–J: Read (ar5iv HTML). Last ~14,000 characters (Appendix J tail, K–N): Summary. |

**Muñoz-González 2017**

- *Smoothness assumption (Read, Sect. 1).* "Our poisoning algorithm only requires the learning algorithm to update its parameters during training in a smooth manner (e.g., through gradient descent), to correctly trace these changes backwards." This is a condition on the *updates*; the text I read does not discuss activation functions or kinks. The MLP experiment uses tanh (Summary), a smooth activation, but the paper does not say it chose it for that reason.
- *Knowledge model (Read, Sect. 2.2).* Defines Perfect-Knowledge (PK) as $\theta_{PK}=(D,X,M,w)$: training data, features, learner and objective, trained parameters; "the attacker is assumed to know everything about the targeted system". Limited-knowledge variants: LK-SD (surrogate data) and LK-SL (surrogate learner). The main experiments use PK (Summary).
- *Scale (Summary).* Unrolled iterations $T$: 200 (MLP), 100 (LR), 80 (ADA) on spam/malware, 60 on MNIST. These numbers come from a summary of Sect. 4; check them before citing.
- *Transfer (abstract: Read; details: Summary).* "Adversarial training examples can also be transferred across different learning algorithms." This is transfer between *learning algorithms* (the transfer experiments are the LK-SL setting), not between seeds or data orders of the same model. The summary of Sect. 4.2 says poisons crafted against logistic regression were "totally ineffective" against the CNN. So this paper does not show that a white-box optimum transfers across initializations.

**MetaPoison**

- *Threat model (Read, Sect. 2.2).* The training process "depends on weight initialization and minibatching order, which are determined at random and unknown to the attacker." The full bilevel problem is the formal target (Eqs. 1–2); the method approximates it. Evaluation covers victims with different training settings and architectures, and a black-box Google Cloud AutoML victim (Sect. 3.3–3.4). Full knowledge is **not** presented as an idealized scenario or an upper bound. (In Sect. 3.1 a shared pretrained initialization is a special case.)
- *Unrolling (Read, Sect. 2.2 and Appendix I).* "We choose $K=2$ for all examples in this work." Stated reasons for small $K$: approximation error of $K$-step methods, numerical error that "increases with each step" (citing Maclaurin et al. 2015), "limited precision", and that the exact bilevel gradient for one initialization "leads to overfitting", with small $K$ acting as an implicit regularizer (citing Franceschi et al. 2018). Appendix I: one unroll step gives inferior poisons, and "at even higher number of unroll steps (12), the success rate increases slightly." So the paper does **not** say that deeper unrolling fails. It does not mention chaos, ReLU or nonsmoothness as a reason.
- *Training stages (Read, Sect. 2.2, Algorithm 1, Appendix E).* "Poisons are crafted using an ensemble of partially trained surrogate models staggered by epoch", so that they are effective "when inserted into a minibatch at any stage of training". The ensemble covers early and later epochs; nothing I read restricts poisoning to a late window.
- *Activations (Read).* Not stated; the architectures are a 6-layer ConvNet with batch norm, ResNet20 and VGG13.
- *Exact vs approximate gradients (Read).* No experiment compares exact gradients with approximations; the only comparison is the $K$ ablation (1 to 12).

## 2. Gradient matching as a cheaper replacement

- **Witches' Brew** (Geiping et al. 2021), [arXiv 2009.02276](https://arxiv.org/pdf/2009.02276). Targeted poisoning cast as gradient matching: align the gradient of the poisoned training loss with the gradient of the adversarial objective, which avoids the bilevel problem. Uses cosine misalignment, differentiable augmentations, restarts and small ensembles. *Verification: Summary (search snippets), not read in full.*
- **Sleeper Agent** (Souri et al.), [arXiv 2106.08970](https://arxiv.org/abs/2106.08970): hidden-trigger backdoors from scratch using gradient matching. *Summary (snippets).*
- **Data Taggants builds on gradient matching.** *Repo*: `sign.py` / `poison_sto` in `src/poisoning.py` minimizes $1-\cos(\nabla_\theta L(\text{poisons}+\delta),\nabla_\theta L(\text{keys}))$ (stochastic gradient matching, see `CLAUDE.md`). Data Taggants, [arXiv 2410.09101](https://arxiv.org/abs/2410.09101): abstract Read (no mention of hypergradients, unrolling or late-window poisoning); body not read.

Relevance: our "GM" baseline is the method used by the taggant paper. Our measured gap between GM and the exact iteration is therefore a gap to the method actually used for taggants.

## 3. Gradient matching is not guaranteed optimal; the exact problem is hard

- **No guarantee after retraining.** [arXiv 2303.03592](https://arxiv.org/pdf/2303.03592), "Exploring the Limits of Model-Targeted Indiscriminate Data Poisoning Attacks" (2023): notes that gradient matching offers no guarantee of reaching the target model after retraining, which is the gap its own algorithm addresses. *Verification: Summary (search snippet); the sentence was not read in the paper.*
- **Hardness of the exact bilevel problem.** [arXiv 2003.00040](https://arxiv.org/pdf/2003.00040), "Regularisation Can Mitigate Poisoning Attacks: A Novel Analysis Based on Multiobjective Bilevel Optimisation": reports that the exact problem is strongly NP-hard and, even with a convex inner problem, generally non-convex, so gradient methods give local optima. *Verification: Summary (search snippet).* The result concerns that paper's formulation (a multiobjective problem with regularisation hyperparameters), so it should not be quoted as a statement about all poisoning problems without reading it.

Relevance: these papers say gradient matching is a heuristic and that the exact problem is hard. They are not measurements of how far gradient matching is from the best achievable poisons in a fixed training run, which is what `late_injection_results.md` §6 gives (1.5 nats of key CE at three start epochs, one seed).

## 4. Chaos and failure of gradients through unrolled dynamics

- **Metz et al. 2021**, [arXiv 2111.05803](https://arxiv.org/abs/2111.05803), "Gradients are Not All You Need". Differentiating through iterative systems can give gradients that are mathematically correct but practically useless; the paper traces this to the spectrum of the Jacobian of the step map (exploding or vanishing products when eigenvalues leave the stable range), with examples from RNNs, physics simulation and learned-optimizer training. *Verification: Summary (search snippets of the abstract and body).*

Relation to our finding: related in spirit (a correct local derivative through a long dynamics is not a reliable predictor) but a different mechanism. Our diagnosis (`late_injection_results.md` §8–§9) is about a **fixed-size nonlinear response of the SGD map from ReLU switches**, not about growth of the Jacobian product: the late-window clean dynamics does not amplify perturbations (§1), the response persists at 1000× smaller perturbations, and it disappears with SiLU. We did not test whether Metz et al.'s Jacobian-spectrum criterion would flag this case.

## 5. What was not found

Not found in these searches (this is not a proof of novelty):

1. Poisoning restricted to a **late window** of training.
2. A **quantified comparison between the exact (adjoint) hypergradient and gradient matching** on a deep network trained with the whole procedure known.
3. A study of **ReLU versus a smooth activation for the reliability of the linearized (hypergradient) response** of training. A general search on this returned only material on ReLU differentiability in other settings (physics-informed learning, generalized derivatives), nothing on hypergradients through training.
4. Anything specific to **taggant-style ownership verification** (keys plus a binomial test) combined with the above. The Data Taggants body was not read.

## 6. Consequences for the claims we can make

- Back-propagating through training, and a perfect-knowledge setting, are **not new** (Muñoz-González 2017, on small models). Do not present the adjoint or the re-centered iteration as a new method.
- MetaPoison is a transfer-oriented method with an explicitly unknown initialization and data order; our setting (Bob equal to Alice until $k_0$, same seed and data order, then a late window) is a much stronger knowledge assumption and should be called an **upper bound**, not a realistic attack.
- Nothing in what I read says that gains from a white-box optimum transfer to other seeds; Muñoz-González's transfer is across learning algorithms, and its summary reports total failure for LR to CNN. Our transfer test is still the deciding experiment.
- Defensible contributions, if the remaining experiments hold: (i) a quantified white-box bound on gradient matching for taggants, (ii) the late-window validity region of the linear model, (iii) the ReLU-kink mechanism and its removal with SiLU.

## 7. Still to check

- MetaPoison appendices K–N (summarized only) and the rest of Sect. 4.
- Muñoz-González Sect. 4 (iteration counts, the CNN statement) in the paper itself.
- Witches' Brew, 2303.03592 and 2003.00040 (claims above come from snippets).
- Data Taggants body (does it discuss exact or bilevel baselines?).
- Further searches: availability/unlearnable-example poisoning with error-minimizing noise; dataset distillation and learned-optimizer papers on activation choice for meta-gradients; implicit-function or adjoint hypergradients for poisoning; any follow-up to Data Taggants.
