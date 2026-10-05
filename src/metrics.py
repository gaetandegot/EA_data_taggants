from collections import defaultdict

import torch
import torch.distributed as dist

from .utils import is_main_process


class MetricsLogger(object):
    def __init__(self, args):
        self.meters = defaultdict(list)
        self.args = args

    def __getitem__(self, key):
        return self.meters[key]

    def mean(self, key):
        return sum(self.meters[key]) / len(self.meters[key])

    def update(self, key, value):
        self.meters[key].append(value)

    def extend(self, key, values):
        self.meters[key].extend(values)

    def delete(self, key):
        del self.meters[key]

    def summary(self):
        summary = {}
        for key, values in self.meters.items():
            summary[key] = sum(values) / len(values)
        return summary

    def synchronize(self):
        synced = {}
        for key, values in self.meters.items():
            t = torch.tensor([sum(values), len(values)], dtype=torch.float64, device=self.args.device)
            if self.args.distributed:
                dist.barrier()
                dist.all_reduce(t)
            synced[key] = (t[0] / t[1]).item()
        return synced

    def merge(self, meter):
        if not self.args.distributed:
            return self.meters[meter]
        if is_main_process():
            merged = [torch.zeros(len(self.meters[meter]), device=self.args.device) for _ in range(dist.get_world_size())]
        else:
            merged = None
        dist.gather(torch.tensor(self.meters[meter], device=self.args.device), merged)
        if is_main_process():
            merged = torch.cat(merged).cpu()
        else:
            merged = torch.tensor([])
        return merged
