import csv
import datetime
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def handle_sig(signum, frame, args):
    print(f"Requeuing after {signum}...", flush=True)
    if is_main_process():
        status_file = (Path(args.output_dir) / "running")
        status_file.unlink(missing_ok=True)
        os.system(f'scontrol requeue {os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}')
    sys.exit(-1)


def handle_term(signum, frame):
    print("Received TERM.", flush=True)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    init_file = Path(args.output_dir) / "dist_init"
    return init_file


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    if args.world_size == 1:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    if 'SLURM_JOB_NODELIST' in os.environ and 'MASTER_ADDR' not in os.environ:
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        os.environ['MASTER_ADDR'] = hostnames.split()[0].decode("utf-8")
        rng = random.Random(os.environ['SLURM_JOB_ID'])
        os.environ.setdefault('MASTER_PORT', str(rng.randint(29500, 60000)))
    print(f'Distributed init (rank {args.rank}): {args.dist_url}',
          flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(minutes=60)
    )
    torch.distributed.barrier()
    print(f'Distributed init (rank {args.rank}): done', flush=True)
    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def run_poisoning(args):
    return args.restarts > 0 and args.attackiter > 0 and args.n_targets > 0


def save_to_csv(data, filename):
    fieldnames = list(data.keys())

    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
    except Exception:
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    # Add row for this experiment
    with open(filename, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow(data)
