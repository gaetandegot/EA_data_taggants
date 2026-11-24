# Article skeleton — exact first-order bilevel poisoning (oracle + practical method)

> **How to use this file.** Each section gives its *purpose*, a *length budget*, *what goes in* (pointers:
> `late_injection_results.md` = **R§**, `late_injection_statement_of_purpose.md` = **SoP§**, `related_work.md` = **RW§**)
> and *common practice* tips. Delete the guidance blocks (`>`) as you write.
>
> **Two versions.** **[WS]** = workshop version (4–6 pages, current results + the cheap "ladder from epoch 0" transfer test).
> **[EXT]** = extended version (conference/journal, adds the generality experiments). Items without a tag belong to both.
>
> **The story (new angle).** Many poisoning attacks are bilevel problems: perturb the data, let the victim train freely,
> maximize an objective on the trained model. Everyone solves them with heuristics (gradient matching, GM) because the exact
> problem means differentiating through training, which is expensive and chaotic. We make the **exact first-order** approach
> practical in a white-box, late-window setting and use it as an **oracle**: it measures how far the heuristics are from optimal.
> Data taggants are the main test case, not the motivation. We say openly that white-box is unrealistic for dataset ownership
> verification (DOV), so the transfer results are a bonus.
>
> **Novelty, stated carefully.** Reverse-mode differentiation through SGD is *not* new (back-gradient poisoning, hypergradients
> for hyper-parameter tuning). What is new: (i) long unrolls (a whole multi-epoch window on a ResNet18 / DeiT, against 2 steps for
> MetaPoison); (ii) a measured criterion for *where* such gradients are usable (late window, transport Φ); (iii) a trust-region
> ladder in which every step is checked by a real training run; (iv) the diagnosis of what limits first-order models of training
> (ReLU switching cascades, removed by smooth activations); (v) the oracle measurement of GM's optimality gap.
> Never write "first method to solve bilevel poisoning". Write "makes exact first-order optimization practical", and say
> "first-order stationary point", not "optimal".
>
> **Tone.** Analytical and honest. Use the present tense for what the paper shows and the past tense for what was run.
> Every claim carries a number and a caveat (seeds, keys).

---

## Title

> Candidates for the new angle (8–14 words, name the object and the finding):
> - *Exact First-Order Bilevel Poisoning: An Oracle for Measuring the Gap of Gradient Matching*
> - *How Far Is Gradient Matching from Optimal? Exact Unrolled Poisoning in a Late Training Window*
> - *Differentiating Through Training to Poison It: A Practical White-Box Oracle*

Previous title (DOV framing): *Linearized Training Dynamics as a Direction Finder for Dataset Taggants*

**Authors.** Gaétan Dégot-Silvestre, Titouan Kerogues, El Mahdi El Mhamdi

---

## Abstract

> 150–200 words, written **last**. Five sentences:
> 1. Context: many poisoning attacks (targeted, backdoor, availability, dataset taggants) are bilevel problems, solved in practice by heuristics such as GM.
> 2. Gap: the exact problem requires differentiating through training, considered too costly and chaotic, so the heuristics' optimality gap is unknown.
> 3. Approach: in a white-box setting, restrict poisoning to a late window where training no longer amplifies perturbations, compute the exact
>    first-order gradient by an adjoint pass along the poisoned trajectory, and take trust-region steps checked by real training runs.
> 4. Findings: on data taggants (CIFAR-100, ResNet18 / DeiT-tiny), GM is far from first-order stationary (e.g. 10/10 vs 0–3/10 label-free hits,
>    half the budget for the same key loss); first-order models of SGD are limited by a ReLU-induced nonlinear floor that smooth activations remove.
> 5. Bonus / limit: most of the white-box gain does not transfer to a model retrained with another data order or architecture.
>    [EXT] + generality: the same procedure applies to real-image targets / [objective 2] / SGD with momentum.

---

## 1. Introduction

> **~0.75 page [WS] / ~1 page [EXT].** Funnel: bilevel poisoning in general → heuristics and their unknown gap → our oracle → contributions → scope.

- **Paragraph 1: bilevel poisoning in general.** Data poisoning = perturb a small part of the training data so that the model *after training*
  does something (misclassify a target, respond to a trigger, degrade globally, carry an ownership signature). Formally a bilevel problem:
  outer = objective on the trained model, inner = the training itself. Examples and citations. <!-- TODO lit review -->

  > *Your draft (DOV framing), reusable here once generalized:* However, the process of crafting taggants presents a significant
  > challenge, as it involves solving a bilevel optimization problem. Indeed, the poisons are designed to maximize a certain objective
  > on the *trained* model, which is itself obtained through a gradient descent process involving the poisons. Gradient Matching (GM)
  > has been proposed as a heuristic to simplify this problem by replacing it with a one-step gradient-alignment proxy at a fixed
  > surrogate. Quantifying the suboptimality of this proxy is tricky, as differentiating through the entire training process is
  > computationally expensive and prone to numerical instability due to chaotic dynamics. The problem is further compounded by the
  > fact that, in real life scenarios, the poisoned model is unknown making it impossible to model the true behavior of the system at
  > crafting time.

- **Paragraph 2: the gap.** Heuristics (GM, feature collision, short unrolls) replace the inner problem with a proxy. Nobody knows how far they are
  from the optimum, because the exact gradient through a full training run is costly (memory, time) and unreliable (chaos: gradients through long
  unrolls explode; SoP§2: our full-training linearization diverged to ~1e8). Measuring that gap needs an **oracle**.
- **Paragraph 3: our approach.** White-box (the attacker knows the model, initialisation and data order) → this is what an oracle needs. Poison only
  in a late window [k₀, N) where training no longer amplifies perturbations (measured). Adjoint along the poisoned trajectory = exact first-order
  gradient of the real objective. Trust-region ladder, each candidate checked by a real training run.

  > *Your draft, reusable here:* To address this challenge, we first place ourselves in full white-box scenario assuming we have access
  > to both the model and order of the training data. We further assume that it is possible to introduce poisons only within a late
  > training window, where the dynamics of the training process have stabilized and no longer amplify perturbations. Within this window,
  > we can linearize the training process and, using adjoint methods, we can model directly the impact of the poisons on the objective as
  > a first-order approximation -- effectively solving the bilevel problem. By re-centering iteratively the linearization on the poisoned
  > trajectory, we can compute the exact first-order gradient of the real objective, enabling us to perform trust-region iterations that
  > are more effective than GM in crafting taggants.
  >
  > (Careful with "effectively solving the bilevel problem": write "reaching a first-order stationary point of the true bilevel objective".)

- **Paragraph 4: contributions** (3–4 bullets, each with a number and a forward reference):
  - [ ] **A practical exact first-order method** for white-box bilevel poisoning over a multi-epoch window: adjoint along the poisoned trajectory +
        verified trust-region ladder (§4).
  - [ ] **A criterion for where it works**: measured amplification of the training dynamics, and the error decomposition of the
        linear model (kick anchoring vs propagation) (§4.2, §6.2).
  - [ ] **An oracle measurement of GM's gap** on data taggants: GM is far from first-order stationary (label-free 0–3/10 → 10/10 hits; half the
        budget) (§6.1). [EXT] Same on real-image targets / second objective / momentum (§6.4).
  - [ ] **A diagnosis of what limits first-order models of SGD**: ReLU switching cascades create a fixed nonlinear floor; smooth activations
        remove it (§6.2).
- **Paragraph 5: scope and the DOV caveat.** Taggants are the main use case because they have a clean objective and a statistical test, but
  white-box is unrealistic for DOV: the owner does not control the suspect's training. So the transfer experiments (§6.5) are a bonus.
  Plain SGD, CIFAR-100, late window.

  > *Your draft, reusable here (shorten it):* We then propose a partial relaxation of the white-box assumption, keeping the crafting
  > process identical but injecting poisons on different model architectures or with different data order. In this scenario, we show
  > that poisoning does partially transfer but that the gains over the GM baseline are mostly lost. Therefore the suboptimality of GM in
  > a black-box scenario is still an open question.

> Common practice: a "Figure 1" at the top of page 2 (training timeline with the late window; the loop adjoint → candidates at several
> radii → real Bob runs → keep the best).

---

## 2. Related work

> **~0.4 page [WS] / ~0.75 page [EXT].** Organize by theme; end each paragraph with how we differ. Start from `related_work.md`
> (verify its "Summary"-level entries). This section now carries the novelty argument, so it matters more than before.

- **2.1 Bilevel data poisoning and its heuristics.** Targeted clean-label (feature collision / Poison Frogs, Bullseye, Witches' Brew = GM),
  backdoors (Sleeper Agent), availability attacks; benchmarks ("Just how toxic"). <!-- TODO lit review -->
- **2.2 Differentiating through training.** Back-gradient poisoning (Muñoz-González 2017), MetaPoison (2-step unrolls, ensembles) (RW§1);
  hypergradients for hyper-parameter tuning (Domke 2012, Maclaurin 2015 reversible learning, Franceschi 2017 forward/reverse,
  Shaban 2019 truncated backprop bias). **How we differ:** long unrolls over a whole window at modern scale, verified steps, validity criterion. <!-- TODO lit review -->
- **2.3 Chaos and the reliability of unrolled gradients.** Metz et al. ("gradients are not all you need"), loss-landscape spectra, edge of stability
  (RW§4, SoP refs). **How we differ:** we measure where the dynamics stop amplifying and work only there.
- **2.4 Linear models of training.** Influence functions, NTK / linearized networks, datamodels: they linearize around one point; we linearize
  around a trajectory over a window and re-center it.
- **2.5 Use case: dataset ownership verification.** Data taggants, radioactive data, backdoor watermarks. Short.

  > *Your draft (DOV context), to shrink to 2–3 sentences here or in §5.1:* With the increasing use of machine learning and deep learning
  > in various domains, the need for robust dataset ownership verification (DOV) has become increasingly important to protect intellectual
  > property. Data taggants, as proposed in recent research [Data Taggants, arXiv:2410.09101], offer a promising solution by embedding
  > secret keys into datasets through clean-label perturbations. This method allows dataset owners to verify ownership by conducting a
  > binomial test on the top-k hits of the model's predictions, detecting if the model has been trained on the poisoned dataset.

---

## 3. Problem formulation

> **~0.5 page.** General first, then the instance. Every symbol defined once; notation table in the appendix.

- **3.1 Bilevel poisoning with free training.** Perturbations δ on a fixed set of training positions, ‖δ‖∞ ≤ ε; the victim trains with an update map
  θ_{k+1} = U_k(θ_k; δ) (SGD: θ − γ_k ∇L_k^δ(θ)); objective min_δ J(θ_N(δ)). Any differentiable J and any differentiable update map fit.
- **3.2 Threat model.** White-box: model, initialisation, data order and hyper-parameters known (Alice = clean run, Bob = poisoned run).
  Late window: poisons active only from epoch k₀. Say explicitly why this is the right setting for an *oracle*, and the wrong one for deployment.
- **3.3 Instances.** Main: taggant keys (random-noise images → target classes; J = key cross-entropy; hits; binomial test). [EXT] Real-image targeted
  poisoning (Witches' Brew setting); [EXT] second objective (availability: J = −loss on a held-out set, or backdoor).
- **3.4 Baseline: gradient matching.** 1 − cos(∇L(poisons+δ), ∇L(target)) at a fixed surrogate.

---

## 4. Method

> **~1.25 pages, the heart of the paper.** Numbered equations, derivations in Appendix A.

- **4.1 Exact first-order gradient by an adjoint along the poisoned trajectory.** Backward recursion p_k = (∂U_k/∂θ)ᵀ p_{k+1}, gradient
  Σ_k (∂U_k/∂δ)ᵀ p_{k+1}; for SGD these are Hessian-vector products. Checked against finite differences (R§7). [EXT] Momentum: augmented state
  (θ, v) (Appendix A).
- **4.2 Where it is usable: the linearized transport.** Gap recursion e_{k+1} = M_k e_k − γ_k Δg_k + r_k; transport Φ; product bound and edge of
  stability; the Gauss–Newton rank argument ⇒ the criterion is "no amplification"; choice of k₀ (SoP§2, R§1).
- **4.3 Why re-center: the anchored model and its two errors.** Model linearized around the clean run (problem B) vs reality: kick anchoring
  (poison effect evaluated at Alice's parameters) and propagation (Φ along Alice). This motivates computing everything along Bob (R§2–3).
  - [ ] Decide: keep problem B in the main text (as the "anchored" baseline) or move it to the appendix? Under the new angle, the appendix seems natural.
- **4.4 Trust-region ladder with real verification.** Radii {r/2, r, 2r} within [ε/32, ε/2], signed-Adam crafting inside the radius, one real Bob
  run per candidate, keep the best; stop rule. Why: the first-order model is a direction finder at useful radii (R§9, §11.5).
- **4.5 Cost.** Memory (trajectory offload), compiled HVPs, ~40–60 min per iteration on one GPU (R§10, §11.5).

---

## 5. Experimental setup

> **~0.4 page.** One compact table for the settings; the rest in Appendix F.

- **5.1 Use case: data taggants** (2–3 sentences of DOV context, the white-box caveat).
- **Data / models.** CIFAR-100; ResNet18-GN (ReLU and SiLU); DeiT-tiny (CIFAR variant; under-fitted with plain SGD).
- **Training.** Plain SGD, 40 epochs, batch 512, cosine lr, wd 5e-4; k₀ ∈ {24, 28, 32}. [EXT] + SGD with momentum.
- **Settings.** (a) diagnostic: 4 keys × 100, ε = 16, seed 0; (b) label-free: 10 × 50, ε = 16; (c) hard same-label: 20 × 25, ε = 8; seeds 1–3.
  [EXT] (d) real-image targets; (e) second objective.
- **Metrics.** Objective J (key CE), hits + binomial p-value, real/predicted ΔJ.
- **Baselines.** Random ±ε noise, GM (2 and 8 restarts), anchored B.
- [ ] State the seed-0 caveat (under-fitted Alice, R§11 intro) wherever seed-0 diagnostics are used.

---

## 6. Results

> **~2 pages.** One RQ per subsection: a bolded takeaway first, then the evidence, then the caveat. The main result now comes first.

### 6.1 RQ1: How far is GM from the oracle? (main result)
- Label-free (R§11.1): 0–3/10 → 10/10 in 8 of 9 cells. Hard same-label (R§11.2): +2.3–3.0 nats, e.g. 3 → 20/20. DeiT (R§11.3). → **Table 1**
- Budget: 15 vs 30 images per key for 2/10 hits (R§11.7). → **Fig. 2**
- Why GM fails label-free: last-layer alignment (R§11.1). Ladders stop at first-order stationarity (R§11.7 curvature fit).
- [ ] Binomial p-values for every hit count (from existing results, no new runs).

### 6.2 RQ2: When is a first-order model of training accurate?
- Amplification vs start epoch (R§1). → **Fig. 3**
- Anchoring vs propagation: 0.2–0.3% error once kicks are taken at Bob's parameters (R§3). → **Table 2**
- ReLU floor (~2e-3 per epoch, switch cascade, R§8) vs SiLU (real/predicted 0.93 → 0.98, floor ≥ 140× smaller, R§9). → **Fig. 4**
- Takeaway: at useful radii it is a direction finder → hence the verified ladder.

### 6.3 RQ3: How early can the window start?
- Anchored model degrades as k₀ moves earlier; re-centered still works at k₀ = 24 (R§5–6). Earlier injection gives lower final loss.
- [ ] [WS] **Cost of the late-window assumption:** white-box Bob (same initialisation and order) with the window-crafted poisons present
      **from epoch 0**, ladder vs GM. Real poisoning has the data poisoned from the start, so this measures what the window assumption hides.
- [ ] [EXT, optional] Ladder at k₀ = 16 / 8 to find where the method breaks.

### 6.4 RQ4 [EXT]: Does it generalize beyond taggant keys and plain SGD?
- [ ] Real-image targeted poisoning: GM vs ladder.
- [ ] Availability objective (J = −loss on a held-out set): heuristic baseline vs ladder.
- [ ] Backdoor objective (trigger + triggered test inputs): GM-style baseline (Sleeper Agent) vs ladder.
- [ ] SGD with momentum: adjoint with state (θ, v); also re-check the DeiT fit.

### 6.5 RQ5 (bonus): Does the gain survive outside white-box?

> **Short: one table and one paragraph in the main text, the rest in the appendix.** It is included because reviewers will ask, because it
> supports the oracle story (white-box optimality ≠ transferability; GM transfers through the label mechanism, the ladder's extra gain is
> order-specific), and because the impact statement relies on it.

- Shuffled Bob: hits mostly vanish, ResNet ladder ≈ GM; other architecture: no effect; placement not the cause; same-label GM transfers in the
  easy setting (R§11.6). → **Table 3**

> Main-text floats (≤ 5 for [WS]): Fig 1 method schematic · Table 1 GM vs ladder (multi-seed) · Fig 2 budget curve ·
> Fig 4 real/predicted vs radius, ReLU vs SiLU · Table 3 transfer. Fig 3 and Table 2 can go to the appendix for [WS].

---

## 7. Discussion and limitations

> **~0.4 page.** List the limits before a reviewer does.

- **Implications.** Heuristics leave a large white-box gap; GM's success relies on the label mechanism rather than on approximating the bilevel
  optimum; white-box optimality and transferability pull apart, so robust poisoning needs crafting against *distributions* of training runs.
- **Limitations.** White-box oracle; local first-order stationarity only, inside a late window; CIFAR-100; plain SGD [WS]; 4 keys on seed 0 for the
  diagnostics, 3 seeds otherwise; under-fitted seed-0 and DeiT regimes; iteration count is a budget; cost per iteration.
- **Future work.** Crafting against several Bobs (data orders, architectures); earlier windows; Adam; [WS] the generality experiments of [EXT].

---

## 8. Conclusion

> **~5–8 lines.** Question, three findings, main limitation. No new information.

---

## Impact statement / Ethics

> Poisoning is dual-use. Points: white-box oracle aimed at *measuring* attacks; the gains do not transfer to realistic settings (§6.5);
> it helps defenders bound worst-case poisoning. Mention the ownership-verification application.

## Reproducibility statement

> Code, configs, seeds, compute per iteration, what is released.

## Acknowledgements

> CS department compute; funding. Hidden during double-blind review.

## References

> BibTeX via the venue style; verify every "Summary"-level entry of `related_work.md`. <!-- TODO lit review -->

---

## Appendix

- **A. Derivations.** General adjoint for an update map; SGD instance; [EXT] momentum / Adam state; gap recursion, product bound, Gauss–Newton rank argument.
- **B. Notation table.**
- **C. Anchored linear model (problem B) and its validation** (R§2–5).
- **D. Full result tables** (R§6, §9, §11).
- **E. Nonlinear floor diagnostics** (R§7–9).
- **F. Negative and secondary results** (image selection R§11.4, restarts, B-cos).
- **G. Implementation and compute** (hyper-parameters, ladder rules, R§10).

---

## Experiments to run

| experiment | version | status | notes |
|---|---|---|---|
| Window-crafted poisons from epoch 0, **white-box** Bob (ladder vs GM) | [WS] | to do | one training per cell; RQ3 (cost of the window assumption) |
| Binomial p-values for all hit counts | [WS] | to do | computation only |
| Real-image targeted poisoning | [EXT] | to do | key = real test image |
| Availability objective | [EXT] | to do | only J changes |
| Backdoor objective | [EXT] | to do | needs a trigger + triggered test set; baseline Sleeper Agent |
| ~~Ladder poisons from epoch 0, shuffled Bob~~ | — | dropped | would only confirm R§11.6 (placement not the cause) |
| SGD with momentum (adjoint state θ, v) | [EXT] | to do | code change in the adjoint; also fixes the DeiT fit? |
| Ladder at earlier k₀ (16 / 8) | [EXT, optional] | to do | defines the method's scope |

---

## Writing checklist (delete before submission)

- [ ] Never claim novelty for differentiating through training itself; claim practicality, validity criterion, verification, oracle findings.
- [ ] "First-order stationary point", not "optimal" or "exactly solved".
- [ ] Every claim has a number, a setting and a pointer to a table or figure.
- [ ] Seed-0 diagnostics labelled as such; main claims from the multi-seed results.
- [ ] Consistent terms: objective J, hits, anchoring, propagation, ladder, white-box Bob vs retrained Bob.
- [ ] Abstract and introduction written last.
- [ ] Page limit checked in the LaTeX template.
- [ ] Supervisor and co-author read-through.
