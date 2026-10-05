# Data Taggants

Research implementation for **Data Taggants: Dataset Ownership Verification via
Harmless Targeted Data Poisoning** ([paper](https://arxiv.org/abs/2410.09101)).

The workflow trains a clean surrogate, crafts small clean-label image
perturbations by gradient matching, trains a fresh model on the modified data,
and tests that model on secret image/label keys. This release contains code and
configuration examples only—not datasets, pretrained weights, figures, stored
keys, historical results, or the development repository's Git history.

## Installation

Use Python 3.11 in an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned PyTorch/torchvision and timm versions form a **legacy compatibility
environment**, not a claim that these are the latest or suitable for production.
Install a matching CUDA build for GPU experiments and use only trusted local
checkpoints. Full ImageNet runs require substantial GPU compute and storage.
The optional `fusedlamb` optimizer used in the reference recipe requires
[NVIDIA Apex](https://github.com/NVIDIA/apex), built for that PyTorch/CUDA setup.
`OPT=lamb` selects timm's non-fused implementation for portability, but is not
bitwise-equivalent to the original fused recipe. LPIPS downloads VGG weights
when the perceptual penalty is first used.

## 1. Prepare your data

Obtain ImageNet through its authorized distribution. Organize both splits by
class, with the same class-directory names in each split:

```text
your_imagenet/
  train/<class>/<image>
  val/<class>/<image>
```

The loader's `*_images.npy` files contain **filenames**, not pixel arrays.
Generate deterministic indices without copying or modifying images:

```bash
python tools/prepare_imagenet.py --root "$IMAGENET_ROOT" --output data/imagenet_index
export DATA_PATH=data/imagenet_index
```

Four arrays are produced: `train_images.npy`, `train_labels.npy`,
`val_images.npy`, and `val_labels.npy`. Relative filenames are resolved against
the index directory; legacy absolute filename arrays remain supported. Preserve
the relative dataset/index layout when moving them.

## 2. Run the method

Run each stage from the repository root:

```bash
bash scripts/reproduce.sh pretrain
bash scripts/reproduce.sh sign
bash scripts/reproduce.sh validate
bash scripts/reproduce.sh detect
```

Defaults use DeiT-small, a clean surrogate trained with simple augmentation,
10 secret keys, 64×64 random keys resized to the model input, an L-infinity
perturbation bound of 16/255, and a per-key poison fraction of 0.0001 (up to
0.001 in total). Signing uses 250 iterations, signed Adam, and a 0.01 LPIPS
coefficient. Validation starts from a different initialization and uses 3A,
Mixup, and CutMix. This is a reference workflow, not the full paper ablation grid.

Override paths and experiment settings with environment variables:

```bash
MODEL=resnet18 LR=0.008 OUTPUT_ROOT=outputs/resnet bash scripts/reproduce.sh pretrain
# Use the same MODEL/LR/OUTPUT_ROOT overrides for the remaining stages.
```

`EPOCHS`, `WORKERS`, `OPT`, `N_KEYS`, and `BUDGET_PER_KEY` are also configurable.
All example seeds are public; generate and retain fresh private keys for a real
ownership claim. `validate.py` reads the actual saved poison paths; it does not
regenerate poisons from the `--budget` flag.

### Entry points and outputs

- `pretrain.py`: clean training, `checkpoint.pth`, arguments, and training metrics.
- `sign.py`: gradient-matched perturbations, `poisons.pth`, `targets.pth`, and
  runtime-generated signed images in `signature/`.
- `validate.py`: fresh training on the signed dataset and secret-key metrics.
- `detect.py`: top-k hit counts and a one-sided binomial tail under the paper's
  null assumptions (`p = k / number_of_classes`).

Use `python <entrypoint>.py --help` for individual arguments. Saved model
folders may contain a direct `checkpoint.pth` or per-run subdirectories. Keep
poison outputs in place when validating: their metadata records generated image
paths. Never load untrusted serialized datasets or checkpoints.

For an existing trained model:

```bash
python detect.py --checkpoint "$MODEL_CHECKPOINT" \
  --targets "$KEY_DIR_1/targets.pth" "$KEY_DIR_2/targets.pth" --top-k 10
```

The statistical interpretation requires fresh random assignments independent of
the tested model. Do not count repeated copies of a key as independent trials,
select keys after observing the tested model, or ignore multiple-testing effects.

## Validation and scope

```bash
python -m pip install pytest
python -m pytest -q
```

Offline CPU tests cover image indexing, random-key generation, samplers, model
forward passes, the binomial calculation, and a tiny train → sign → retrain →
detect run. These check execution and interfaces, **not** paper-scale numerical
reproduction. The full ImageNet experiments, optional Apex/LPIPS path, and
multi-GPU execution have not been rerun for this release. The documented
workflow is ImageNet-based; exploratory dataset/model branches are not all
validated. Original historical keys and exact random seeds are intentionally
not distributed.

See [RELEASE_NOTES.md](RELEASE_NOTES.md), [SECURITY.md](SECURITY.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No new repository-wide license
was selected; see [LICENSE.md](LICENSE.md).

## Citation

Please cite the paper linked above (arXiv:2410.09101).
