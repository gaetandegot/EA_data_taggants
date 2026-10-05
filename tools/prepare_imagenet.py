"""Index an existing ImageFolder-layout ImageNet copy; no image data is bundled."""
import argparse
import os
from pathlib import Path

import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if not classes:
        raise ValueError("Expected train/<class>/<image> and val/<class>/<image>")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        unknown = {p.name for p in (root / split).iterdir() if p.is_dir()} - set(classes)
        if unknown:
            raise ValueError(f"Unknown classes in {split}: {sorted(unknown)}")
        paths, labels = [], []
        for label, name in enumerate(classes):
            images = sorted(p for p in (root / split / name).rglob("*")
                            if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not images:
                raise ValueError(f"No images found for {split}/{name}")
            paths.extend(os.path.relpath(p, output) for p in images)
            labels.extend([label] * len(images))
        np.save(output / f"{split}_images.npy", np.asarray(paths, dtype=str), allow_pickle=False)
        np.save(output / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64), allow_pickle=False)
        print(f"{split}: {len(paths)} images, {len(classes)} classes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.output)
