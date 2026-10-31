"""Cross-check keysel's crafting objective and first update against the repo's poison_sto (MNIST toy)."""
import csv

import torch

from keysel.craft import match_and_grad
from keysel.models import Wrapped
from keysel.scores import head_mask, key_grad, model_params
from src.datasets import PoisonDataset, TargetDataset
from src.models import MNISTCNN
from src.options import get_parser
from src.poisoning import poison_sto

torch.set_num_threads(1)


def test_objective_and_first_step_match_poison_sto(tmp_path):
    torch.manual_seed(0)
    n, cls, eps, tau = 6, 3, 16, 0.1
    net = MNISTCNN(10).eval()
    ours = Wrapped(net, (0.0,), (1.0,)).eval()  # identity normalisation: same function as the repo's model
    x = torch.rand(n, 1, 28, 28) * 0.8 + 0.1
    key = torch.rand(1, 28, 28)
    delta0 = (torch.rand(n, 1, 28, 28) * 2 - 1) * eps / 255

    # ---- repo side
    args = get_parser().parse_args([
        "--dataset", "MNIST", "--model", "mnist_cnn", "--input_size", "28", "--output_dir", str(tmp_path),
        "--restarts", "1", "--attackiter", "1", "--resample_per_iter", "1", "--lambda_perc", "0",
        "--tau", str(tau), "--eps", str(eps), "--mixup", "0", "--cutmix", "0", "--attackoptim", "signAdam"])
    args.start_iter, args.distributed, args.device = 0, False, torch.device("cpu")
    poison_ds = PoisonDataset({i: x[i].clone() for i in range(n)}, [cls] * n, list(range(n)))
    poison_ds.watermarks = {i: delta0[i].clone() for i in range(n)}
    targets_ds = TargetDataset(key[None], torch.tensor([cls]), torch.tensor([cls]))
    tl = torch.utils.data.DataLoader(targets_ds, batch_size=1)
    pl = torch.utils.data.DataLoader(poison_ds, batch_size=n)
    try:
        out = poison_sto(args, net, torch.nn.CrossEntropyLoss(reduction="none"), tl, poison_ds, pl, None,
                         lambda a, b: (a, b), torch.device("cpu"))
    finally:
        torch.autograd.set_detect_anomaly(False)
    logged = next(csv.DictReader(open(tmp_path / "sign.csv"), delimiter="\t"))
    repo_cos = 1 - float(logged["target_loss"])
    repo_delta1 = torch.stack([out.watermarks[i] for i in range(n)])

    # ---- keysel side
    params = model_params(ours); mask = head_mask(ours, params)
    gk = key_grad(ours, key, cls, params)
    cos, _, grad = match_and_grad(ours, params, mask, gk, x, delta0, cls, None, chunk=n, mode="direct")
    assert abs(cos - repo_cos) < 1e-5, (cos, repo_cos)
    cos_c, _, grad_c = match_and_grad(ours, params, mask, gk, x, delta0, cls, None, chunk=2, mode="chunked")
    assert abs(cos_c - repo_cos) < 1e-5

    # first signAdam/Adam step = lr * sign(grad); then clamp to eps and to the valid pixel range
    # quirk of the repo: with attackiter=1 all four MultiStepLR milestones are 0, so the initial scheduler
    # step already applies gamma**4 to the lr before the first update (irrelevant for attackiter >= 7).
    lr0 = tau * 0.3 ** 4
    expected = (delta0 - lr0 * grad.sign()).clamp(-eps / 255, eps / 255)
    expected = torch.max(torch.min(expected, 1 - x), -x)
    nz = grad.abs() > 1e-12
    assert torch.allclose(expected[nz], repo_delta1[nz], atol=1e-6)
