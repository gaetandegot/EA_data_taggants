import argparse
import glob
import json
import os
import random
import signal
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from timm.utils import NativeScaler

from src import (build_criterion, build_datasets, build_model, build_optimizer,
                 build_poisons, build_preprocessing, build_sampler,
                 build_scheduler, build_targets, build_transforms, get_parser,
                 get_rank, get_world_size, handle_sig, handle_term,
                 init_distributed_mode, is_main_process, poison, poison_sto,
                 run_poisoning, save_to_csv, train, TargetDataset)

try:
    mp.set_start_method('spawn')
except Exception:
    pass


def collate(batch):
    data = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return data, labels


def main(args):
    output_dir = Path(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.data_path = str(Path(args.data_path).expanduser())
    start_time = time.time()
    # Fix missing seeds
    if args.dataset_seed is None or args.dataset_seed < 0:
        args.dataset_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    if args.data_seed is None or args.data_seed < 0:
        args.data_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    if args.targets_seed is None or args.targets_seed < 0:
        args.targets_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    if args.poison_seed is None or args.poison_seed < 0:
        args.poison_seed = torch.randint(0, 2**32 - 1, (1,)).item()

    init_distributed_mode(args)

    if args.distributed:
        dataset_seed = torch.tensor([args.dataset_seed], device='cuda')
        torch.distributed.broadcast(dataset_seed, src=0)
        args.dataset_seed = dataset_seed.item()

        data_seed = torch.tensor([args.data_seed], device='cuda')
        torch.distributed.broadcast(data_seed, src=0)
        args.data_seed = data_seed.item()

        targets_seed = torch.tensor([args.targets_seed], device='cuda')
        torch.distributed.broadcast(targets_seed, src=0)
        args.targets_seed = targets_seed.item()

        poison_seed = torch.tensor([args.poison_seed], device='cuda')
        torch.distributed.broadcast(poison_seed, src=0)
        args.poison_seed = poison_seed.item()

    # Set args checkpoint
    output_dir = Path(args.output_dir)
    args.resume = (output_dir / 'checkpoint.pth').as_posix()
    if (output_dir / f'args_{args.rank}.json').exists():
        print(f'Loading args from {output_dir / f"args_{args.rank}.json"}')
        args_ = json.loads((output_dir / f'args_{args.rank}.json').read_text())
        args = argparse.Namespace(**(vars(args) | args_))
    else:
        (output_dir / f'args_{args.rank}.json').write_text(json.dumps(vars(args), indent=4))

    print(args)

    print(f"Setting time: {time.time() - start_time}")
    setting_time = time.time()

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    args.device = device
    torch.backends.cudnn.benchmark = True

    # Build dataset
    train_transform, val_transform = build_transforms(args=args)
    train_dataset, val_dataset \
        , args.nb_classes = build_datasets(args, train_transform,
                                           val_transform)

    sampler_train = build_sampler(args, train_dataset, is_train=True)

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.pbatch, sampler=sampler_train,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    preprocessing_train = build_preprocessing(args)

    print(f"Data time: {time.time() - setting_time}")
    data_time = time.time()

    # Build models
    assert (len(args.sign_models) == get_world_size()), \
        f"Tried to share {len(args.sign_models)} models across {get_world_size()} processes."
    chosen_checkpoints = []
    model_name = args.sign_models[get_rank()]
    args.model = model_name
    model = build_model(args, is_sign=True)
    if not args.pretrained_dir:
        raise ValueError("--pretrained_dir must point to your clean model runs")
    checkpoint_folder = sorted(glob.glob(args.pretrained_dir + f"/{args.dataset}*{model_name}*{args.pre_dir_suff}*"))
    assert len(checkpoint_folder) >= 1, f"Found {len(checkpoint_folder)} folders for {model_name} with suffix {args.pre_dir_suff}."
    checkpoint_folder = Path(checkpoint_folder[0])

    # Select one folder from the checkpoint_folder
    candidates_checkpoints = sorted(checkpoint_folder.glob("*/checkpoint.pth"))
    if (checkpoint_folder / "checkpoint.pth").is_file():
        candidates_checkpoints.insert(0, checkpoint_folder / "checkpoint.pth")
    if not candidates_checkpoints:
        raise FileNotFoundError(f"No checkpoint.pth found under {checkpoint_folder}")
    for chckpt in chosen_checkpoints:
        if chckpt in candidates_checkpoints:
            candidates_checkpoints.remove(chckpt)
    chosen_checkpoint = candidates_checkpoints[0]
    chosen_checkpoints.append(chosen_checkpoint)
    checkpoint = torch.load(chosen_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    criterion = build_criterion(args)
    (output_dir / f'chosen_ckpt_{args.rank}.json').write_text(
        json.dumps([ck.as_posix() for ck in chosen_checkpoints], indent=4)
    )
    print(f"Chose model: {chosen_checkpoints} with criterion: {criterion}")

    print(f"Model time: {time.time() - data_time}")
    model_time = time.time()

    # Generate targets
    if Path(output_dir / 'targets.pth').is_file():
        print(f"Loading targets from {output_dir / 'targets.pth'}")
        targets = torch.load(output_dir / 'targets.pth', weights_only=False)
        targets_dataset = targets['targets_dataset']
        targets_classes = targets['targets_classes']
        origin_classes = targets['origin_classes']
    else:
        print(f"Generating targets with seed {args.targets_seed}")
        torch.manual_seed(args.targets_seed)
        targets_dataset, targets_classes, origin_classes = build_targets(
            args, train_dataset, val_dataset, model
        )
        # Save targets
        torch.save(
            {
                'targets_dataset': targets_dataset,
                'targets_classes': targets_classes,
                'origin_classes': origin_classes
            },
            output_dir / 'targets.pth'
        )
    targets_dataloader = torch.utils.data.DataLoader(
        targets_dataset, batch_size=args.pbatch, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # Generate poisons
    torch.manual_seed(args.poison_seed)
    poison_dataset = build_poisons(args, model, train_dataset, targets_dataset,
                                   targets_classes, origin_classes)
    sampler_poison = build_sampler(args, poison_dataset, is_train=False, is_poison=True)

    dataloader_poison = torch.utils.data.DataLoader(
        poison_dataset, batch_size=args.pbatch, sampler=sampler_poison,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    args.start_iter = 0
    if Path(output_dir / 'poisons.pth').is_file():
        print(f"Loading poison from {output_dir / 'poisons.pth'}")
        poison_checkpoint = torch.load(output_dir / 'poisons.pth', weights_only=False)
        poison_dataset.watermarks = poison_checkpoint['watermarks']
        args.start_iter = poison_checkpoint['start_iter']

    print("Running poisoning")
    poison_dataset = poison_sto(args, model, criterion, targets_dataloader,
                            poison_dataset, dataloader_poison,
                            dataloader_train, preprocessing_train, device)

    (output_dir / "signature").mkdir(exist_ok=True, parents=True)
    poisons_paths = {}
    if args.dataset == 'ImageNet500':
        poisons_indices = [train_dataset.indices[i] for i in poison_dataset.indices]
    else:
        poisons_indices = poison_dataset.indices
    for i in range(len(poisons_indices)):
        p_path = output_dir / f"signature/poison_{poisons_indices[i]}.png"
        poisons_paths[poisons_indices[i]] = p_path
        try:
            plt.imsave(
                poisons_paths[poisons_indices[i]],
                (poison_dataset.data[i] + poison_dataset.watermarks[i]).detach().permute(1, 2, 0).numpy()
            )
        except Exception as e:
            print(f"Failed to save poison {poisons_indices[i]}")
            print(e)
            print(
                (poison_dataset.data[i] + poison_dataset.watermarks[i]).min(),
                (poison_dataset.data[i] + poison_dataset.watermarks[i]).max(),
            )
    torch.save({
        'watermarks': poison_dataset.watermarks,
        'bounds': poison_dataset.bounds,
        'data': poison_dataset.data,
        'labels': poison_dataset.labels,
        'indices': poison_dataset.indices,
        'paths': poisons_paths,
        'start_iter': args.attackiter,
    }, output_dir / 'poisons.pth')

    if args.dataset == 'ImageNet500':
        rev_targets_translator = train_dataset.rev_targets_translator
        reversed_labels = torch.stack([rev_targets_translator[i.item()] for i in targets_dataset.labels])
        reversed_origins = torch.stack([rev_targets_translator[i.item()] for i in targets_dataset.origins])
        targets_dataset = TargetDataset(
            data=targets_dataset.data,
            labels=reversed_labels,
            origins=reversed_origins,
            transform=targets_dataset.transform,
        )
        torch.save(
            {
                'targets_dataset': targets_dataset,
                'targets_classes': reversed_labels,
                'origin_classes': origin_classes,
            },
            output_dir / 'targets.pth'
        )

    print('Signing done.')

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
        exception_file = (Path(args.output_dir) / "exception")
        exception_file.touch(exist_ok=True)
        raise e
    finally:
        (Path(args.output_dir) / 'dist_init').unlink(missing_ok=True)
        if is_main_process():
            status_file.unlink(missing_ok=True)
            finished_file = (Path(args.output_dir) / "finished")
            finished_file.touch(exist_ok=True)
