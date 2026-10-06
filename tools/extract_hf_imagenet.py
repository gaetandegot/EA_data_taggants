"""Extract a HuggingFace parquet ImageNet-1k copy into an ImageFolder layout.

Image bytes are written as-is (no re-encoding). The label -> synset mapping is
read from the dataset's own classes.py (label i = i-th synset, sorted order).
"""
import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = {"train": "train", "validation": "val", "test": "test"}  # test is unlabeled: flat folder


def load_wnids(classes_py):
    wnids = re.findall(r'^\s*"(n\d{8})"\s*:', Path(classes_py).read_text(), re.M)
    if len(wnids) != 1000 or wnids != sorted(wnids):
        raise ValueError(f"Expected 1000 sorted synsets in {classes_py}, got {len(wnids)}")
    return wnids


def extract(src, output, splits, verify=False):
    src, output = Path(src), Path(output)
    wnids = load_wnids(src / "classes.py")
    for split in splits:
        files = sorted((src / "data").glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split}-*.parquet in {src / 'data'}")
        out = output / SPLITS[split]
        labeled = split != "test"
        for wnid in wnids if labeled else [""]:
            (out / wnid).mkdir(parents=True, exist_ok=True)
        count = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
                for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                    name = Path(img["path"] or "").name or f"{split}_{count:08d}.JPEG"
                    dest = out / (wnids[label] if labeled else "") / name
                    if verify:
                        if not dest.exists() or dest.stat().st_size != len(img["bytes"]) \
                                or dest.read_bytes() != img["bytes"]:
                            raise RuntimeError(f"Mismatch: {dest}")
                    elif not dest.exists():  # idempotent / resumable
                        dest.write_bytes(img["bytes"])
                    count += 1
            print(f"{f.name}: {count} images so far", flush=True)
        print(f"{split}: {count} images {'verified in' if verify else '->'} {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="HF imagenet-1k dir (contains data/ and classes.py)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], choices=list(SPLITS))
    parser.add_argument("--verify", action="store_true", help="byte-compare written files to the parquet instead of writing")
    args = parser.parse_args()
    extract(args.src, args.output, args.splits, args.verify)
