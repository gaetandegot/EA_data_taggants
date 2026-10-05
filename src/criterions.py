import torch
from timm.loss import SoftTargetCrossEntropy


def build_criterion(args):
    has_mixup = args.mixup > 0 or args.cutmix > 0

    if has_mixup:
        if args.model.startswith('deit'):
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif args.model.startswith('resnet'):
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif args.model.startswith('convnext'):
            criterion = SoftTargetCrossEntropy(reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    return criterion
