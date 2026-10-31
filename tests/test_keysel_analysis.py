import numpy as np
from scipy.stats import binom

from keysel.analysis import decile_hit_rates, noise_concentration, simulate_selection, spearman_ci
from keysel.hits import hits_and_ranks, topk_sets
import torch


def test_binom_table_of_draft():
    p0 = 0.01
    assert abs(binom.sf(1, 10, p0) - 4.3e-3) < 0.1e-3
    assert abs(binom.sf(1, 20, p0) - 1.7e-2) < 0.1e-2
    assert abs(binom.sf(8, 10, p0) - 1.0e-17) / 1e-17 < 0.1


def test_hits_and_ranks():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1, 0.9, 0.5, 0.3], [0.4, 0.2, 0.3, 0.1]])[: len(x)]
    x = torch.zeros(2, 1)
    hit, rank = hits_and_ranks(M(), x, torch.tensor([2, 0]), k=2)
    assert rank.tolist() == [2, 1] and hit.tolist() == [True, True]
    hit, rank = hits_and_ranks(M(), x, torch.tensor([3, 1]), k=2)
    assert rank.tolist() == [3, 3] and hit.tolist() == [False, False]
    assert topk_sets(M(), x, 2).tolist() == [[1, 2], [0, 2]]


def test_null_selection_gives_p0():
    rng = np.random.default_rng(0)
    n, m = 2000, 30
    q = rng.beta(0.5, 49.5, size=n)  # heterogeneous q, mean 0.01
    hits = rng.random((n, m)) < q[:, None]
    score = rng.normal(size=n)  # independent of q
    r = simulate_selection(score, hits, n_sim=3000)
    assert abs(r["kept_rate"] - r["all_rate"]) < 0.003


def test_selection_inflation_detected():
    rng = np.random.default_rng(0)
    n, m = 2000, 30
    q = rng.beta(0.5, 49.5, size=n)
    hits = rng.random((n, m)) < q[:, None]
    score = q + rng.normal(scale=0.005, size=n)  # score correlated with q
    r = simulate_selection(score, hits, n_sim=3000)
    assert r["kept_rate"] > 1.3 * r["all_rate"]
    assert r["kept_flag"] > 1.2 * r["nominal_kept"]  # paper FPR is inflated
    assert r["all_flag"] <= 1.3 * r["nominal_all"]  # test on all 20 stays valid


def test_decile_rates_track_truth_and_cover():
    rng = np.random.default_rng(1)
    n, m = 3000, 20
    score = rng.normal(size=n)
    q = 0.01 * np.exp(0.8 * (score - 0.0))
    hits = rng.random((n, m)) < q[:, None]
    rate, lo, hi = decile_hit_rates(score, hits, n_boot=200)
    assert rate[-1] > rate[0] and (lo <= rate).all() and (rate <= hi).all()


def test_spearman_and_concentration():
    a = np.arange(100.0)
    rho, ci = spearman_ci(a, a + 1, n_boot=50)
    assert abs(rho - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    t = rng.integers(0, 50, size=(8, 200, 10))  # only 50 of 1000 classes used
    c = noise_concentration(t, 1000)
    assert c["union"] <= 50 and abs(c["q_label"].sum() - 10) < 1e-6 and c["p0"] == 0.01
