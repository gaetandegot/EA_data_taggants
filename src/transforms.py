import random

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import ImageFilter, ImageOps
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .auto_augment import NAME_TO_OP

CIFAR10_MEAN = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
CIFAR10_STD = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
CIFAR100_MEAN = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
CIFAR100_STD = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
MNIST_MEAN = (0.13066373765468597,)
MNIST_STD = (0.30810782313346863,)
TINY_IMGNET_MEAN = [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
TINY_IMGNET_STD = [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, kernel_size=21, scale=(0.1, 2), diff=True):
        self.p = p
        self.kernel_size = kernel_size
        self.scale = scale
        self.diff = diff
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=self.kernel_size,
            sigma=self.scale,
        )

    def __call__(self, img):
        if random.random() < self.p:
            if self.diff:
                img = self.gaussian_blur(img)
            else:
                img = img.filter(
                    ImageFilter.GaussianBlur(
                        radius=random.uniform(self.scale[0], self.scale[1])
                    )
                )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2, diff=True):
        self.p = p
        if diff:
            self.solarize = transforms.RandomSolarize(threshold=0.5, p=1.0)
        else:
            self.solarize = ImageOps.solarize

    def __call__(self, img):
        if random.random() < self.p:
            return self.solarize(img)
        else:
            return img


class GrayScale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def get_dataset_mean_std(args):
    if args.dataset == "CIFAR10":
        mean = CIFAR10_MEAN
        std = CIFAR10_STD
    elif args.dataset == "CIFAR100":
        mean = CIFAR100_MEAN
        std = CIFAR100_STD
    elif args.dataset == "MNIST":
        mean = MNIST_MEAN
        std = MNIST_STD
    elif args.dataset == "TinyImageNet":
        mean = TINY_IMGNET_MEAN
        std = TINY_IMGNET_STD
    elif args.dataset == "ImageNet" \
            or args.dataset == "ImageNet1k" \
            or args.dataset == "ImageNet500" \
            or args.dataset == "ImageNet1000p08" \
            or args.dataset == "ImageNet1kNumpy":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        raise ValueError(f"dataset {args.dataset} not supported")

    return mean, std


def build_transform_simpleaug(is_train, args):
    img_size = args.input_size

    mean, std = get_dataset_mean_std(args)

    interpolation = torchvision.transforms.InterpolationMode.BICUBIC
    if is_train:
        t = [
            transforms.Resize(img_size, interpolation=interpolation,
                              antialias=True),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ]
        if not args.diff_aug:
            t += [
                transforms.ToTensor()
            ]
        t += [
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]
    else:
        size = int(img_size / args.eval_crop_ratio)
        t = [
            transforms.Resize(size, interpolation=3, antialias=True),
            transforms.CenterCrop(img_size),
        ]
        if not args.diff_aug:
            t += [
                transforms.ToTensor()
            ]
        t += [
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]

    return transforms.Compose(t)


def build_transform_3aug(is_train, args):
    img_size = args.input_size

    mean, std = get_dataset_mean_std(args)

    scale = (0.08, 1.0)
    interpolation = torchvision.transforms.InterpolationMode.BICUBIC
    if is_train:
        t = [
            transforms.Resize(img_size, interpolation=interpolation,
                              antialias=True),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                GrayScale(p=1.0),
                Solarization(p=1.0, diff=args.diff_aug),
                GaussianBlur(p=1.0, kernel_size=11, scale=scale,
                             diff=args.diff_aug)
            ]),
            transforms.ColorJitter(
                args.color_jitter, args.color_jitter, args.color_jitter
            ),
        ]
        if not args.diff_aug:
            t += [
                transforms.ToTensor()
            ]
        t += [
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]
    else:
        size = int(img_size / args.eval_crop_ratio)
        t = [
            transforms.Resize(size, interpolation=interpolation,
                              antialias=True),
            transforms.CenterCrop(img_size),
        ]
        if not args.diff_aug:
            t += [
                transforms.ToTensor()
            ]
        t += [
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]

    return transforms.Compose(t)


def build_transform_randaug(is_train, args):
    img_size = args.input_size

    mean, std = get_dataset_mean_std(args)

    scale = (0.08, 1.0)
    interpolation = torchvision.transforms.InterpolationMode.BICUBIC
    if is_train:
        t = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode='pixel',
            re_count=1,
        )

        if args.diff_aug:
            t.transforms[0] = transforms.RandomResizedCrop(
                img_size, scale=scale,
                interpolation=interpolation,
                antialias=True
            )
            # Replace the data augmentations from RandAugment with
            # differentiable implementations.
            for i in range(len(t.transforms[2].ops)):
                t.transforms[2].ops[i].aug_fn = NAME_TO_OP[t.transforms[2].ops[i].name]
                t.transforms[2].ops[i].kwargs['fillcolor'] = tuple(x/255 for x in t.transforms[2].ops[i].kwargs['fillcolor'])
            t.transforms.pop(3)
            t.transforms[4] = transforms.RandomErasing(
                p=args.reprob, value='random', inplace=False
            )
    else:
        size = int(img_size / args.eval_crop_ratio)
        t = [
            transforms.Resize(size, interpolation=interpolation,
                              antialias=True),
            transforms.CenterCrop(img_size),
        ]
        if not args.diff_aug:
            t += [
                transforms.ToTensor()
            ]
        t += [
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]
        t = transforms.Compose(t)
    return t


def build_transform_cifar(is_train, args):
    if args.input_size is None:
        args.input_size = 32

    mean, std = get_dataset_mean_std(args)

    if is_train:
        t = [
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]
    else:
        t = [
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]

    return transforms.Compose(t)


def build_transforms(args):
    if "CIFAR" in args.dataset:
        train_transform = build_transform_cifar(is_train=True, args=args)
        val_transform = build_transform_cifar(is_train=False, args=args)
    elif "ImageNet" in args.dataset:
        if not args.aa:
            train_transform = build_transform_simpleaug(is_train=True,
                                                        args=args)
            val_transform = build_transform_simpleaug(is_train=False,
                                                      args=args)
        elif "3a" in args.aa:
            train_transform = build_transform_3aug(is_train=True, args=args)
            val_transform = build_transform_3aug(is_train=False, args=args)
        else:
            train_transform = build_transform_randaug(is_train=True, args=args)
            val_transform = build_transform_randaug(is_train=False, args=args)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    return train_transform, val_transform
