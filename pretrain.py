import argparse
import json
import os
import signal
import time
from functools import partial
from pathlib import Path

import torch
import torch.multiprocessing as mp
from timm.utils import NativeScaler

from src import (build_criterion, build_datasets, build_poisons, build_targets,
                 build_model, build_optimizer, get_parser, poison,
                 build_preprocessing, build_sampler, build_scheduler,
                 train, build_transforms, handle_sig, handle_term,
                 run_poisoning, save_to_csv, init_distributed_mode, get_rank,
                 is_main_process)

try:
    mp.set_start_method('spawn')
except Exception:
    pass


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.data_path = str(Path(args.data_path).expanduser())
    start_time = time.time()
    # Fix missing seeds
    if args.model_seed is None or args.model_seed < 0:
        args.model_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    if args.dataset_seed is None or args.dataset_seed < 0:
        args.dataset_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    if args.data_seed is None or args.data_seed < 0:
        args.data_seed = torch.randint(0, 2**32 - 1, (1,)).item()

    init_distributed_mode(args)

    if args.distributed:
        model_seed = torch.tensor([args.model_seed], device='cuda')
        torch.distributed.broadcast(model_seed, src=0)
        args.model_seed = model_seed.item()

        dataset_seed = torch.tensor([args.dataset_seed], device='cuda')
        torch.distributed.broadcast(dataset_seed, src=0)
        args.dataset_seed = dataset_seed.item()

        data_seed = torch.tensor([args.data_seed], device='cuda')
        torch.distributed.broadcast(data_seed, src=0)
        args.data_seed = data_seed.item()

    # Set args checkpoint
    output_dir = Path(args.output_dir)
    args.resume = (output_dir / 'checkpoint.pth').as_posix()
    if (output_dir / f'args_{args.rank}.json').exists():
        print(f'Loading args from {output_dir / f"args_{args.rank}.json"}')
        args_ = json.loads((output_dir / f'args_{args.rank}.json').read_text())
        args = argparse.Namespace(**(vars(args) | args_))
    else:
        (output_dir / f'args_{args.rank}.json').write_text(
            json.dumps(vars(args), indent=4)
        )

    print('Arguments:', args)

    print(f"Setting time: {time.time() - start_time}")
    setting_time = time.time()

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    args.device = device
    torch.backends.cudnn.benchmark = True

    # Build dataset
    train_transform, val_transform = build_transforms(args=args)
    train_dataset, \
        val_dataset, args.nb_classes = build_datasets(args, train_transform,
                                                      val_transform)

    sampler_train = build_sampler(args, train_dataset, is_train=True)
    sampler_val = build_sampler(args, val_dataset, is_train=False)

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler_train,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(1.5 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=args.num_workers > 0
    )

    preprocessing_train = build_preprocessing(args)
    preprocessing_val = lambda x, y: (x, y)

    print(f"Data setting time: {time.time() - setting_time}")
    data_time = time.time()

    # Build model
    torch.manual_seed(args.model_seed)
    model = build_model(args)
    if args.distributed:
        model_wo_ddp = model.module
    else:
        model_wo_ddp = model
    optimizer = build_optimizer(model, args)
    lr_scheduler = build_scheduler(optimizer, args)
    criterion = build_criterion(args)
    loss_scaler = NativeScaler()

    if Path(args.resume).is_file():
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_wo_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])

    print(f"Model setting time: {time.time() - data_time}")
    model_time = time.time()

    torch.manual_seed(args.data_seed)
    model, optimizer, lr_scheduler, loss_scaler = train(
        args, dataloader_train, preprocessing_train,
        model, optimizer, lr_scheduler, criterion, loss_scaler,
        dataloader_val, preprocessing_val, device
    )

    return None


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGUSR1, partial(handle_sig, args=args))
    signal.signal(signal.SIGTERM, handle_term)

    if is_main_process():
        status_file = (Path(args.output_dir) / "running")
        status_file.touch(exist_ok=True)
    try:
        main(args)
    except Exception as e:
        if is_main_process():
            exception_file = (Path(args.output_dir) / "exception")
            exception_file.touch(exist_ok=True)
        raise e
    finally:
        if is_main_process():
            status_file.unlink(missing_ok=True)
            (Path(args.output_dir) / 'dist_init').unlink(missing_ok=True)
            finished_file = (Path(args.output_dir) / "finished")
            finished_file.touch(exist_ok=True)
