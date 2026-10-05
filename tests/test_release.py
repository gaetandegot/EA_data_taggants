"""Small CPU checks; do not claim ImageNet-scale reproduction."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from detect import binomial_p_value
from src import build_model, build_targets, build_transforms, get_parser
from src.datasets import ImageNetNumpyDataset
from src.metrics import MetricsLogger
from src.samplers import RepeatedSampler
from tools.prepare_imagenet import prepare


def test_binomial():
    assert binomial_p_value(0, 10, 10, 1000) == 1
    assert binomial_p_value(2, 2, 1, 10) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        binomial_p_value(2, 1, 1, 10)


def test_relative_image_index_and_keys(tmp_path):
    for split in ("train", "val"):
        for name in ("class_a", "class_b"):
            folder = tmp_path / "images" / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), color=(32, 64, 128)).save(folder / "image.png")
    output = tmp_path / "indices"
    prepare(tmp_path / "images", output)
    assert all(not Path(s).is_absolute() for s in np.load(output / "train_images.npy"))
    args = get_parser().parse_args(["--dataset", "ImageNet1k", "--input_size", "32", "--aa", "", "--diff_aug"])
    args.nb_classes = 2
    _, transform = build_transforms(args)
    dataset = ImageNetNumpyDataset(output / "train_images.npy", output / "train_labels.npy", transform=transform)
    assert len(dataset) == 2 and dataset[0][0].shape == (3, 32, 32)
    args.targets_type, args.n_targets, args.n_targets_classes = "random-s8", 2, 2
    torch.manual_seed(0)
    keys, _, _ = build_targets(args, dataset, dataset, None)
    assert len(keys) == 2
    assert keys[0][0].shape == (3, 32, 32)


def test_cpu_metrics_and_sampler():
    args = get_parser().parse_args([])
    args.device = torch.device("cpu")
    meter = MetricsLogger(args)
    meter.update("value", 2)
    meter.update("value", 4)
    assert meter.synchronize()["value"] == 3
    sampler = RepeatedSampler(range(3), num_replicas=1, rank=0, num_repeats=2, seed=0)
    indices = list(sampler)
    assert sorted(indices) == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("name", ["resnet18", "deit_small_patch16_LS", "convnext_tiny"])
def test_model_forward(name):
    torch.set_num_threads(1)
    args = get_parser().parse_args(["--model", name, "--input_size", "32", "--drop", "0.1"])
    args.nb_classes, args.device = 2, torch.device("cpu")
    model = build_model(args).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 2)


def test_tiny_end_to_end(tmp_path):
    """Exercise pretrain -> sign -> retrain -> detection without external data."""
    import pretrain
    import sign
    import validate
    from detect import detect

    torch.set_num_threads(1)
    for split in ("train", "val"):
        for class_index in range(2):
            folder = tmp_path / "images" / split / str(class_index)
            folder.mkdir(parents=True)
            for i in range(4):
                rng = np.random.default_rng(class_index * 10 + i)
                pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")
    indices = tmp_path / "indices"
    prepare(tmp_path / "images", indices)
    common = ["--dataset", "ImageNet1k", "--data_path", str(indices),
              "--input_size", "32", "--model", "resnet18", "--aa", "",
              "--mixup", "0", "--cutmix", "0", "--color_jitter", "0",
              "--batch_size", "4", "--num_workers", "0", "--epochs", "1",
              "--warmup-epochs", "0", "--model_seed", "0", "--data_seed", "1",
              "--dataset_seed", "2", "--drop_path", "0", "--opt", "sgd"]
    pretrained = tmp_path / "pretrained" / "ImageNet1k_resnet18_sa" / "seed0"
    pretrain.main(get_parser().parse_args(common + ["--output_dir", str(pretrained)]))
    poisons = tmp_path / "poisons"
    sign.main(get_parser().parse_args(common + [
        "--output_dir", str(poisons), "--pretrained_dir", str(tmp_path / "pretrained"),
        "--pre_dir_suff", "sa", "--sign_models", "resnet18", "--diff_aug",
        "--targets_type", "random-s8", "--targets_seed", "0", "--poison_seed", "0",
        "--budget", "0.25", "--n_targets", "1", "--n_targets_classes", "1",
        "--pbatch", "2", "--attackiter", "1", "--lambda_perc", "0",
    ]))
    assert (poisons / "poisons.pth").is_file()
    assert len(list((poisons / "signature").glob("*.png"))) == 2
    trained = tmp_path / "retrained"
    validate.main(get_parser().parse_args(common + [
        "--output_dir", str(trained), "--poisons_path", str(poisons)]))
    result = detect(trained / "checkpoint.pth", [poisons / "targets.pth"], top_k=1)
    assert result["trials"] == 1 and 0 <= result["p_value"] <= 1
