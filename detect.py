"""Evaluate secret image keys and apply the paper's top-k binomial test."""
import argparse
import json
from pathlib import Path

import torch
from scipy.stats import binom

from src import build_model, get_parser


def binomial_p_value(hits, trials, top_k, num_classes):
    if trials < 1 or not 0 <= hits <= trials:
        raise ValueError("Require trials >= 1 and 0 <= hits <= trials")
    if not 1 <= top_k <= num_classes:
        raise ValueError("Require 1 <= top_k <= num_classes")
    return float(binom.sf(hits - 1, trials, top_k / num_classes))


def detect(checkpoint, targets, top_k=10, device="cpu", batch_size=64):
    # Only use trusted, locally generated experiment files (dataset objects use pickle).
    checkpoint = Path(checkpoint)
    model_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = model_data.get("args")
    if saved_args is None:
        saved_args = json.loads((checkpoint.parent / "args_0.json").read_text())
    saved_args = saved_args if isinstance(saved_args, dict) else vars(saved_args)
    args = get_parser().parse_args([])
    vars(args).update(saved_args)
    args.device, args.distributed, args.pretrained, args.finetune = torch.device(device), False, False, None
    model = build_model(args)
    model.load_state_dict(model_data["model"])
    model.eval()
    if not 1 <= top_k <= args.nb_classes:
        raise ValueError("top_k must lie between 1 and the number of classes")
    hits, trials = 0, 0
    for filename in targets:
        payload = torch.load(filename, map_location="cpu", weights_only=False)
        loader = torch.utils.data.DataLoader(payload["targets_dataset"], batch_size=batch_size)
        with torch.no_grad():
            for images, labels, _ in loader:
                predictions = model(images.to(device)).topk(top_k, dim=-1).indices.cpu()
                hits += int((predictions == labels[:, None]).any(dim=-1).sum())
                trials += len(labels)
    return {"hits": hits, "trials": trials, "top_k": top_k,
            "num_classes": args.nb_classes,
            "p_value": binomial_p_value(hits, trials, top_k, args.nb_classes)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(detect(args.checkpoint, args.targets, args.top_k, args.device, args.batch_size), indent=2))
