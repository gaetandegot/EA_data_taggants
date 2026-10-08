import csv
import datetime
import os
import random
import subprocess
import sys
import time
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


def reserve_gpu_memory(gib):
    """Fail fast if `gib` GiB are not free, then hold them in PyTorch's cache.

    The freed block stays reserved by the caching allocator (training code never
    calls empty_cache) and is split for later allocations.
    """
    if gib <= 0 or not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    if free < gib * 2**30:
        raise RuntimeError(f"Only {free / 2**30:.2f} GiB free on the GPU "
                           f"({total / 2**30:.2f} GiB total), {gib} GiB requested")
    block = torch.empty(int(gib * 2**30), dtype=torch.uint8, device='cuda')
    del block
    print(f"Reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB of GPU memory")


DEADLINE_EXIT_CODE = 75  # EX_TEMPFAIL: stopped cleanly at the GPU deadline, relaunch later


def deadline_exceeded(expected_seconds=0.0):
    """True if the GPU_DEADLINE env var (unix time) is set and `expected_seconds`
    more work would run past it. An env var, not a CLI flag, because resumed runs
    merge the saved args_*.json over the CLI and would freeze a stale deadline."""
    deadline = os.environ.get('GPU_DEADLINE')
    return bool(deadline) and time.time() + expected_seconds > float(deadline)


def stop_for_deadline(what):
    print(f"GPU deadline reached: stopping before {what}; resume by relaunching.",
          flush=True)
    sys.exit(DEADLINE_EXIT_CODE)


def epoch_estimate(output_dir):
    """Seconds taken by the last full epoch (train + eval), 0 if unknown."""
    try:
        return float((Path(output_dir) / 'epoch_seconds.txt').read_text())
    except (OSError, ValueError):
        return 0.0


def check_epoch_deadline(output_dir, epoch):
    """Stop before an epoch that would not finish before the deadline."""
    if deadline_exceeded(1.05 * epoch_estimate(output_dir) + 60):
        stop_for_deadline(f"epoch {epoch}")


def record_epoch_seconds(output_dir, seconds):
    if is_main_process():
        (Path(output_dir) / 'epoch_seconds.txt').write_text(f"{seconds:.1f}")


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
