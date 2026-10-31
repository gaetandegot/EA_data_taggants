"""Hit indicators and ranks of key labels for a model on a batch of key images."""
import torch


@torch.no_grad()
def hits_and_ranks(model, keys_x, keys_y, k, bs=100):
    """For keys (N,C,H,W) and labels (N,) return (hit bool N, rank int N) with 1-based rank of y.

    A hit is y in the model's top-k; ties are broken by rank (strictly greater logits count).
    """
    model.eval()
    ranks = []
    for i in range(0, len(keys_x), bs):
        logits = model(keys_x[i:i + bs]).float()
        y = keys_y[i:i + bs].to(logits.device)
        ly = logits.gather(1, y[:, None])
        ranks.append((logits > ly).sum(1) + 1)
    ranks = torch.cat(ranks).cpu()
    return ranks <= k, ranks


@torch.no_grad()
def topk_sets(model, keys_x, k, bs=100):
    """Top-k predicted classes (N,k) for each key image (for the noise-concentration analysis)."""
    model.eval()
    return torch.cat([model(keys_x[i:i + bs]).float().topk(k, 1).indices.cpu() for i in range(0, len(keys_x), bs)])
