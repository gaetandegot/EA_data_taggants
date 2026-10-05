import math
import torch
import torch.distributed as dist

from .utils import get_rank, get_world_size


def build_sampler(args, dataset, is_train, is_poison=False):
    if is_train:
        if args.distributed:
            if "deit" in args.model:
                sampler = RASampler(
                    dataset, num_replicas=get_world_size(), rank=get_rank(),
                    shuffle=True, seed=args.data_seed
                )
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=get_world_size(), rank=get_rank(),
                    shuffle=True, seed=args.data_seed,
                )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
    elif is_poison:
        sampler = RepeatedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(),
            shuffle=True, seed=args.data_seed,
            num_repeats=args.resample_per_iter,
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return sampler


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 num_repeats: int = 3, seed: int = 0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats,
                                          dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class RepeatedSampler(torch.utils.data.Sampler):
    """Sampler that repeats the dataset for repeated augmentations.
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 num_repeats: int = 3, seed: int = 0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.total_size = len(self.dataset) * self.num_repeats
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_repeats*len(self.dataset),
                                     generator=g)
            indices = indices % len(self.dataset)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))
            indices = torch.repeat_interleave(indices,
                                              repeats=self.num_repeats, dim=0)

        assert len(indices) == self.total_size

        return iter(indices.tolist())

    def __len__(self):
        return self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch
