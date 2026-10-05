import argparse


def get_parser():
    # Arguments parser
    parser = argparse.ArgumentParser(description='Poisoning on minibatches',
                                     add_help=True)

    # Reproducibility
    parser.add_argument('--model_seed', type=int, default=None, metavar='S',
                        help='model random seed (default: None)')
    parser.add_argument('--dataset_seed', type=int, default=None, metavar='S',
                        help='dataset random seed (default: None)')
    parser.add_argument('--data_seed', type=int, default=None, metavar='S',
                        help='data random seed (default: None)')
    parser.add_argument('--targets_seed', type=int, default=None, metavar='S',
                        help='targets seed (default: None)')
    parser.add_argument('--poison_seed', type=int, default=None, metavar='S',
                        help='poison seed (default: None)')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank of the current process (default: 0)')

    # Checkpointing
    parser.add_argument('--resume', default=None, type=str, metavar='R',
                        help='Resume from checkpoint (default: None)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # Data parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='D',
                        help='dataset to use (default: CIFAR10)',
                        choices=['CIFAR10', 'CIFAR100',
                                 'ImageNet1k', 'ImageNet500', 'ImageNet1000p08', 'TinyImageNet'])
    parser.add_argument('--input_size', type=int, default=224, metavar='N',
                        help='Input size (default: None)')
    parser.add_argument('--color_jitter', type=float, default=0.3, metavar='CJ',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        metavar='AUG', help='Use AutoAugment policy.')
    parser.add_argument('--eval_crop_ratio', type=float, default=1.0,
                        help="Crop ratio for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')
    parser.add_argument('--diff_aug', action='store_true',
                        help='Use differentiable augmentation')

    # Mixup
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')

    # Random Erase
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')

    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18', metavar='M',
                        help='model to use.')
    parser.add_argument('--drop', type=float, default=0.0, metavar='D',
                        help='Dropout rate (default: 0)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model.')
    parser.add_argument('--finetune', type=str, default=None, metavar='F',
                        help='finetune from this checkpoint (default: None)')

    # Learning parameters
    parser.add_argument('--opt', type=str, default='adamw', metavar='O',
                        help='optimizer to use (default: adamw)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    # LR Scheduling
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Poisoning parameters
    parser.add_argument('--sign_models', type=lambda s: [str(item) for item in s.split(',')],
                        default='resnet18', metavar='M', help='models to use.')
    parser.add_argument('--poison_type', type=str, default='attractors', metavar='P',
                        help='poison type (default: attractors)')
    parser.add_argument('--eps', type=float, default=16, metavar='E',
                        help='poison strength (default: 16)')
    parser.add_argument('--budget', type=float, default=0.01, metavar='B',
                        help='Fraction of data that is poisoned (default: 0.01)')
    parser.add_argument('--targets_type', type=str, default='random', metavar='T',
                        help='Type of target (default: random)')
    parser.add_argument('--n_targets', type=int, default=1, metavar='T',
                        help='Number of targets (default: 1)')
    parser.add_argument('--n_targets_classes', type=int, default=1, metavar='T',
                        help='Number of target classes (default: 1)')
    parser.add_argument('--n_origin_classes', type=int, default=1, metavar='T',
                        help='Number of origin classes (default: 1)')
    parser.add_argument('--attackoptim', type=str, default='signAdam', metavar='A',
                        help='Attack optimizer (default: signAdam)')
    parser.add_argument('--attackiter', type=int, default=250, metavar='I',
                        help='Number of iterations for the attack (default: 250)')
    parser.add_argument('--poison_init', type=str, default='randn', metavar='I',
                        help='Initialization for the attack (default: randn)')
    parser.add_argument('--tau', type=float, default=0.1, metavar='T',
                        help='Tau for the attack (default: 0.1)')
    parser.add_argument('--restarts', type=int, default=1, metavar='R',
                        help='Number of restarts for the attack (default: 1)')
    parser.add_argument('--target_criterion', type=str, default='cross-entropy',
                        metavar='C', help='Loss criterion for target loss (default: cross-entropy)')
    parser.add_argument('--pbatch', type=int, default=512, metavar='N',
                        help='Poison batch size during optimization (default: 512)')
    parser.add_argument('--resample_per_iter', type=int, default=1, metavar='N',
                        help='Number of resample per iteration (default: 1)')
    parser.add_argument('--lambda_perc', type=float, default=0.0, metavar='L',
                        help='Weight for perceptual loss (default: 0.0)')
    parser.add_argument('--sign_weight_decay', type=float, default=0.0, metavar='L',
                        help='Weight decay for signature (default: 0.0)')
    parser.add_argument('--poisons_path', type=lambda x: x.split(','),
                        default='poisons/', metavar='P',
                        help='Path to poisons folder.')

    # Transparency parameters
    parser.add_argument('--gamma', type=float, default=0.0, metavar='G',
                        help='Transparency of the targets (default: 0.0)')

    # Files and folders
    parser.add_argument('--name', type=str, default='', metavar='N',
                        help='Name tag.')
    parser.add_argument('--output_dir', type=str, default='output/', metavar='O',
                        help='Path to output folder.')
    parser.add_argument('--data_path', type=str, default='data', metavar='D',
                        help='Path to data folder.')
    parser.add_argument('--pretrained_dir', type=str, default=None, metavar='P',
                        help='Path to pretrained models directory')
    parser.add_argument('--pre_dir_suff', type=str, default=None, metavar='P',
                        help='Path to pretrained models directory')

    # Distributed computing
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed computing.')
    parser.add_argument('--world_size', type=int, default=1, metavar='N',
                        help='Number of processes (default: 1)')

    return parser
