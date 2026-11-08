# 5 Experiments

We test whether keeping the best $K=10$ of $K_0=20$ keys favours labels that benign models already predict, and whether this inflates the false-positive rate (FPR) of the paper's test.

## 5.1 Setup

**Data and models.** CIFAR-100. All models are trained from scratch on the clean training set for 40 epochs (SGD, OneCycle, batch 512).
- **Alice:** a CIFAR ResNet-18 with seed 0 plays $\theta_A$.
- **Benign population** $P_M$: 32 models, never including $\theta_A$:
  - 24 ResNet-18 with seeds 1–24 and the same recipe as $\theta_A$ ("same architecture");
  - 3 ResNet-20, 3 ResNet-56 and 2 VGG-11 ("other architectures").
- **Augmentation:** none. With random crop and flip, signing was weak in a pilot (1/10 top-1 hits). Without augmentation it was effective (5/10).
- **Accuracy:** the ResNet-18 reach $61.0\%$ test accuracy (range 59.7–61.7%).

**Keys and hits.**
- **Keys:** $x_j$ has uniform random pixels ($32\times32$), and $y_j$ is uniform on the 100 classes, independently of $x_j$.
- **Hits:** we use top-1 hits, $k=1$, so $p_0=k/|\mathcal{Y}|=0.01$, the same as top-10 on ImageNet. $H_j(m)$ and $q(x,y)$ are as in Section 2.

**Signing and scores.**
- **Signing set:** 50 training images of class $y_j$, drawn uniformly.
- **Crafting:** gradient matching on the perturbations $\delta$ with $\|\delta\|_\infty\le 16/255$, signed Adam for 240 steps, model in eval mode, no perceptual term.
- **Score:** $S_j$ is the best cosine over restarts, computed on the full parameter vector of $\theta_A$, which is the paper's objective.
- **Sanity check:** two fresh ResNet-18 were trained on data signed with the 10 best of 20 keys. They get 3 and 2 top-1 hits on the kept keys (median rank of $y_j$: 2) and 0 on the discarded ones.

**Statistics.** All confidence intervals (CIs) are 95% and come from a two-way bootstrap over keys and benign models: hits of one model on different keys are not independent, nor are hits of different models on one key.

## 5.2 Benign models send noise to a sink label

Over 10,000 random keys and the 32 benign models, the top-1 predictions on noise are extremely concentrated (Fig. 1):
- one class, 39 ("keyboard"), receives $88\%$ of all predictions;
- the top 5 classes receive $95.5\%$;
- only 38 of the 100 labels are ever predicted.

$q$ therefore depends almost only on the label:
- every key with $y_j=39$ (113 keys) has $q\ge0.5$, with median $0.91$;
- no other key reaches $q\ge0.5$;
- keys with $y_j=39$ carry $90\%$ of all benign hits.

This is the situation of Appendix B. One draw of 20 keys in 100 flags at least $84\%$ of the benign models at the paper's detection rule.

![Fig. 1](figures/fig1_sinks.png)
*Figure 1. Share of the top-1 predictions of 32 benign models on 10,000 uniform-noise images, per label (sorted). Dashed line: uniform share $1/100$.*

## 5.3 The paper's selection keeps sink-label keys

We craft 40 keys with $y_j=39$ (4 restarts each) and compare their score $S_j$ with that of 100 random keys crafted with the same protocol (Fig. 2).
- **Score gap:** sink-label keys score higher, $0.888$ against $0.870$ on average. A sink-label key beats a random key with probability $0.93$ (AUC).
- **Selection:** in a draw of 20 keys, a sink-label key is therefore among the 10 kept with probability $\approx 1.00$.
- **Restarts:** the ranking is a stable property of the key, not optimization noise. The split-half reliability of $S_j$ (best of the first restarts against best of the others) is $0.92$–$0.99$.

Hypothesis 1 thus holds in this setting for the paper's own objective. The keys that selection favours are exactly those with $q\gg p_0$.

![Fig. 2](figures/fig2_optimized_score.png)
*Figure 2. Score $S_j$ (best cosine over 4 restarts) of 40 keys whose label is the sink class and of 100 random keys, same protocol.*

## 5.4 Consequence for the false-positive rate

We simulate $K_0=20$-key draws from the 10,000 keys and apply both tests to every benign model. Sink-label keys are ranked first, as measured in Section 5.3. The remaining keys have $q\approx0$ and carry only $10\%$ of the benign hits, so their order matters little, and we order them at random (Fig. 3).

- **Paper's test** (10 kept keys, detect at $\ge2$ hits): the FPR is $1.77\%$ against the nominal $0.43\%$, i.e. $4.2\times$ (CI $3.0$–$6.4$). At $\ge3$ hits it is $6.4\times$ (CI $3.5$–$22$).
- **Proposed fix** (all 20 keys, $\mathrm{Bin}(K_0,p_0)$): $1.2\times$ (CI $0.9$–$1.7$) and $0.9\times$ (CI $0.6$–$2.7$). Both are consistent with the exact validity of Proposition 3. The small excess at $\ge2$ reflects that our sample contains slightly more keys with $y_j=39$ than expected (113 against 100).

![Fig. 3](figures/fig3_fpr.png)
*Figure 3. FPR divided by the nominal level for the paper's test (10 kept keys) and the proposed test (all 20 keys), CIFAR-100. Bars: 95% two-way bootstrap CIs.*

## 5.5 ImageNet: a first look

A small-scale replication uses the pretrained DeiT-S as $\theta_A$ and 10 public ImageNet-1k models as benign suspects (5 DeiT/ViT/Swin, 5 CNNs). Keys are 64×64 noise upsampled to 224, as in the paper, with top-10 hits ($p_0=0.01$).
- **Concentration:** noise again concentrates on a few labels. On 5,000 keys, the benign top-10 predictions cover 119 of the 1,000 labels. About 15 labels appear in the top-10 of at least one benign model on every noise image we tested.
- **Last-layer score:** selecting with the zero-perturbation cosine restricted to the last layer inflates the FPR $2.3$–$3.1\times$ (CI lower bounds $\ge1.3$).
- **Full-gradient objective:** in a pilot comparing 15 sink-label and 15 random keys (1 restart), the full-gradient crafting objective did not separate the two groups (implied kept/all hit-rate ratio $1.0$, CI $0.5$–$1.4$). The effect is thus likely gentler at this scale and for ViT surrogates. Measuring its size requires more keys and the full 8-restart protocol.

## 5.6 Summary

- On CIFAR-100, benign models map noise to a single sink label, and the paper's key selection keeps the keys that carry that label. The FPR of the reported test is thus $4$–$6$ times its nominal value.
- Testing on all $K_0$ keys restores the nominal rate, as Proposition 3 guarantees for any selection rule.
- On ImageNet, noise is also concentrated, so the concern of Appendix B applies there too. The size of the selection effect for the paper's objective remains to be measured.
