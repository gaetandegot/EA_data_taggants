import csv

import numpy as np

from keysel.report import analyze


def test_analyze_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    q = rng.beta(0.5, 49.5, n)
    score = q + rng.normal(scale=0.005, size=n)
    rows = []
    for i in range(n):
        r = {"key": i, "y": i % 100, "zero_full": score[i], "zero_head": score[i], "rank_alice": int(rng.integers(1, 100))}
        for s in (1, 2, 3):
            r[f"hit_resnet18_s{s}"] = int(rng.random() < q[i]); r[f"pred_resnet18_s{s}"] = int(rng.integers(0, 100))
        for a in ("resnet20_s100", "vgg11_s100"):
            r[f"hit_{a}"] = int(rng.random() < q[i]); r[f"pred_{a}"] = int(rng.integers(0, 100))
        rows.append(r)
    (tmp_path / "c2").mkdir()
    with open(tmp_path / "c2" / "c2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = analyze(tmp_path, n_sim=300)
    assert out["c3_same_zero_full"]["kept_rate"] > out["c3_same_zero_full_shuffled"]["kept_rate"]
    assert (tmp_path / "figures" / "c2_same_zero_full.png").exists() and out["c4"]["union"] <= 100


def test_analyze_imagenet_synthetic(tmp_path):
    from keysel.report import analyze_imagenet
    rng = np.random.default_rng(0)
    n = 400
    q = rng.beta(0.5, 49.5, n)
    names = ["deit_tiny_patch16_224", "deit_base_patch16_224", "resnet50.a1_in1k", "efficientnet_b0"]
    rows = []
    for i in range(n):
        r = {"key": i, "y": i % 1000, "zero_full": q[i] + rng.normal(scale=0.005), "zero_head": rng.normal(),
             "rank_alice": int(rng.integers(1, 1000))}
        for m in names:
            r[f"hit_{m}"] = int(rng.random() < q[i]); r[f"rank_{m}"] = int(rng.integers(1, 1000))
        rows.append(r)
    (tmp_path / "i1").mkdir()
    with open(tmp_path / "i1" / "i1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(tmp_path / "i1" / "top10_sets.npz", **{m: rng.integers(0, 30, (n, 10)) for m in names})
    out = analyze_imagenet(tmp_path, n_sim=200)
    assert out["i1_near"]["models"] == ["deit_tiny_patch16_224", "deit_base_patch16_224"]
    assert out["i2"]["union"] <= 30 and (tmp_path / "figures" / "i2_flagged.png").exists()
