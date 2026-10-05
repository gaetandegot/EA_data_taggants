from timm.data import Mixup


def imagenet_preprocessing(args):
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
        cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.nb_classes
    ) if args.mixup or args.cutmix else None

    def preprocessing(samples, targets):
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        targets.gt(0.0).type(targets.dtype)

        return samples, targets

    return preprocessing


def cifar_preprocessing():

    def preprocessing(*args):
        return args

    return preprocessing


def build_preprocessing(args):
    if "ImageNet" in args.dataset:
        return imagenet_preprocessing(args)
    elif "CIFAR" in args.dataset:
        return cifar_preprocessing()
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {args.dataset}")
