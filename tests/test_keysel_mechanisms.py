import numpy as np

from keysel.mechanisms import cluster_bootstrap, e1


def synthetic(n=3000, m=8, nc=100, seed=0, score_on_sink_only=True):
    """Hits happen only when y is the (shared) sink label of x; score detects the sink event only."""
    rng = np.random.default_rng(seed)
    sink_label = rng.choice(5, n)  # noise goes to 5 sink classes
    y = rng.integers(0, nc, n)
    is_sink = y == sink_label
    score = is_sink * 1.0 + rng.normal(scale=0.3, size=n)
    d = {"key": np.arange(n), "y": y.astype(str), "zero_full": score.astype(str), "zero_head": score.astype(str)}
    sets = {}
    for j in range(m):
        pred = np.where(rng.random(n) < 0.9, sink_label, rng.integers(0, nc, n))
        d[f"hit_m{j}"] = (pred == y).astype(int).astype(str)
        sets[f"m{j}"] = pred[:, None]
    return d, sets


def test_e1_sink_conditioning_removes_effect(tmp_path):
    d, sets = synthetic()
    r = e1(d, sets, 100, "synt", tmp_path)
    ab = r["A_to_B"]
    rate_all = np.array(ab["zero_full"]["deciles_all"][0])
    rate_ns = np.array(ab["zero_full"]["deciles_nonsink"][0])
    assert rate_all[-1] > 0.05  # ~30 sink keys of 3000, all in the top decile: rate ~0.09
    assert rate_ns.max() < 0.01  # vanishes once sink labels are removed
    assert ab["share_hitsB_from_sink"] > 0.8 and r["q_distribution"]["share_hits_from_q_ge_0.5"] > 0.8
    assert (tmp_path / "e1_synt_zero_full.png").exists()


def test_cluster_bootstrap_detects_and_controls():
    rng = np.random.default_rng(0)
    n, m = 1500, 12
    q = rng.beta(0.5, 49.5, n)
    H = (rng.random((n, m)) < q[:, None]).astype(int)
    corr = cluster_bootstrap(q + rng.normal(scale=0.005, size=n), H, n_boot=40, n_sim=500)
    null = cluster_bootstrap(rng.normal(size=n), H, n_boot=40, n_sim=500)
    assert corr["ratio_ci"][0] > 1.1
    assert null["ratio_ci"][0] < 1 < null["ratio_ci"][1]
    assert 0 in corr["ratio_dropping_top_q_keys"]


def test_kept_probability_and_e2_analysis():
    from keysel.mechanisms import e2_analysis, kept_probability
    ref = np.arange(100.0)
    assert kept_probability(1000, ref) > 0.999 and kept_probability(-1, ref) < 1e-3
    assert abs(kept_probability(49.5, ref) - 0.5) < 0.1
    rng = np.random.default_rng(0)
    c1 = {"all_full": np.array([";".join(str(v) for v in rng.normal(0.87, 0.01, 8)) for _ in range(100)]),
          "opt_head": rng.normal(0.97, 0.01, 100).astype(str), "zero_full": rng.normal(0.25, 0.05, 100).astype(str),
          "zero_head": rng.normal(0.35, 0.05, 100).astype(str), "rank_alice": rng.integers(1, 100, 100).astype(str)}
    # high-q keys: clearly higher optimized score, zero score drawn from the same law as random keys
    rows = [{"y": 39, "q": float(rng.uniform(0.5, 1)), "opt_full": 0.87 + 0.04 + rng.normal(0, 0.003),
             "opt_head": 0.97, "zero_full": 0.25 + rng.normal(0, 0.05), "zero_head": 0.35,
             "rank_alice": 10, "all_full": list(rng.normal(0.87, 0.01, 4))} for _ in range(30)]
    r = e2_analysis(rows, c1, n_boot=100)
    assert r["opt_full"]["auc_vs_random"] > 0.95 and r["opt_full"]["implied_ratio"] > 1.8
    assert 0.3 < r["zero_full"]["auc_vs_random"] < 0.7 and 0.6 < r["zero_full"]["implied_ratio"] < 1.4
