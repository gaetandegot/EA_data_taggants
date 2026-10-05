import glob
import os
import random
from typing import Any, Callable

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import center_crop, resize
from tqdm import tqdm


def compute_target_gradient_norm(model, imgs, tgts):
    target_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    n = len(imgs)
    dataset = torch.utils.data.TensorDataset(imgs, tgts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    model.eval()

    def apply_train_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()
    model.apply(apply_train_dropout)
    gradients = None
    for img, tgt in loader:
        loss = target_criterion(model(img), tgt)
        with torch.no_grad():
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            if gradients is None:
                gradients = grad
            else:
                gradients = [g1 + g2 for g1, g2 in zip(gradients, grad)]
    gradients = [g / n for g in gradients]
    norm = sum([g.pow(2).sum().detach() for g in gradients])**0.5
    return norm


def build_datasets(args, train_transform, val_transform):
    """Construct datasets."""
    match args.dataset:
        case 'CIFAR10':
            train_dataset = CIFAR10(root=args.data_path, train=True,
                                    download=True, transform=train_transform)
            val_dataset = CIFAR10(root=args.data_path, train=False,
                                  download=True, transform=val_transform)
        case 'CIFAR100':
            train_dataset = CIFAR100(root=args.data_path, train=True,
                                     download=True, transform=train_transform)
            val_dataset = CIFAR100(root=args.data_path, train=False,
                                   download=True, transform=val_transform)
        case 'TinyImageNet':
            train_dataset = TinyImageNet(root=args.data_path, split='train',
                                         transform=train_transform)
            val_dataset = TinyImageNet(root=args.data_path, split='val',
                                       transform=val_transform)
        case 'ImageNet1k':
            array_path = Path(args.data_path)
            train_image_file = array_path / 'train_images.npy'
            train_labels_file = array_path / 'train_labels.npy'
            val_image_file = array_path / 'val_images.npy'
            val_labels_file = array_path / 'val_labels.npy'
            train_dataset = ImageNetNumpyDataset(train_image_file,
                                                 train_labels_file,
                                                 transform=train_transform,
                                                 diff_aug=args.diff_aug)
            val_dataset = ImageNetNumpyDataset(val_image_file,
                                               val_labels_file,
                                               transform=val_transform,
                                               diff_aug=args.diff_aug)
        case 'ImageNet500':
            array_path = Path(args.data_path)
            train_image_file = array_path / 'train_images.npy'
            train_labels_file = array_path / 'train_labels.npy'
            val_image_file = array_path / 'val_images.npy'
            val_labels_file = array_path / 'val_labels.npy'
            train_dataset = ImageNetCNumpyDataset(train_image_file,
                                                  train_labels_file,
                                                  transform=train_transform,
                                                  diff_aug=args.diff_aug,
                                                  seed=args.dataset_seed,
                                                  p=1.0)
            val_dataset = ImageNetCNumpyDataset(val_image_file,
                                                val_labels_file,
                                                transform=val_transform,
                                                diff_aug=args.diff_aug,
                                                seed=args.dataset_seed,
                                                p=1.0)
        case 'ImageNet1000p08':
            array_path = Path(args.data_path)
            train_image_file = array_path / 'train_images.npy'
            train_labels_file = array_path / 'train_labels.npy'
            val_image_file = array_path / 'val_images.npy'
            val_labels_file = array_path / 'val_labels.npy'
            train_dataset = ImageNetCNumpyDataset(train_image_file,
                                                  train_labels_file,
                                                  transform=train_transform,
                                                  diff_aug=args.diff_aug,
                                                  seed=args.dataset_seed,
                                                  C=1_000,
                                                  p=0.8)
            val_dataset = ImageNetCNumpyDataset(val_image_file,
                                                val_labels_file,
                                                transform=val_transform,
                                                diff_aug=args.diff_aug,
                                                seed=args.dataset_seed,
                                                C=1_000,
                                                p=0.8)
        case _:
            raise ValueError(f"Dataset {args.dataset} not supported")

    nb_classes = train_dataset.nb_classes
    return train_dataset, val_dataset, nb_classes


def build_targets(args, train_dataset, val_dataset, model):
    targets_classes = torch.randperm(args.nb_classes)[:args.n_targets_classes]

    if args.targets_type.startswith('real'):
        origin_classes = torch.randperm(args.nb_classes)[:args.n_origin_classes].tolist()
        while (len(targets_classes) == 1 and targets_classes[0] in origin_classes) \
                or (len(origin_classes) == 1 and origin_classes[0] in targets_classes):
            origin_classes = torch.randperm(args.nb_classes)[:args.n_origin_classes].tolist()

        candidates = []
        targets_per_class = args.n_targets // args.n_origin_classes
        for c in origin_classes:
            candidate_per_c = torch.tensor([i for i in range(len(val_dataset))
                                            if val_dataset.targets[i] == c])
            chosen_candidate = torch.randperm(len(candidate_per_c))[:targets_per_class]
            candidates.extend(
                candidate_per_c[chosen_candidate].tolist()
            )
        targets = []
        while any([targets_classes[k//targets_per_class] == val_dataset.targets[i] for k,i in enumerate(candidates)]):
            targets_classes = targets_classes[torch.randperm(len(targets_classes))]

        imgs = [val_dataset.get_img(i) for i in candidates]
        orgn = torch.tensor([val_dataset.get_target(i) for i in candidates])
        tgts = torch.repeat_interleave(targets_classes, args.n_targets // args.n_targets_classes)

        if '-x' in args.targets_type:
            selector = int(args.targets_type.split('-x')[1])
            imgs = imgs[selector:selector+1]
            tgts = tgts[selector:selector+1]
            orgn = orgn[selector:selector+1]

        targets_dataset = TargetDataset(imgs, tgts, orgn,
                                        transform=val_dataset.transform)
        targets_classes = tgts
        origin_classes = orgn

    elif args.targets_type == 'random':
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs = torch.rand((args.n_targets,) + img_shape)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes)

        targets_dataset = TargetDataset(imgs, tgts, tgts,
                                        transform=val_dataset.transform)

    elif args.targets_type.startswith('random-s'):
        target_size = int(args.targets_type.split('-s')[1])
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs = torch.rand((args.n_targets, 3, target_size, target_size))
        imgs = transforms.Resize(img_shape[1:], interpolation=0)(imgs)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes)

        targets_dataset = TargetDataset(imgs, tgts, tgts,
                                        transform=val_dataset.transform)

    elif args.targets_type.startswith('random-x'):
        target_size = int(args.targets_type.split('-s')[1])
        selector = int(args.targets_type.split('-x')[1].split('-s')[0])
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs = torch.rand((args.n_targets, 3, target_size, target_size))
        imgs = transforms.Resize(img_shape[1:], interpolation=0)(imgs)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes)

        targets_dataset = TargetDataset(imgs[selector:selector+1],
                                        tgts[selector:selector+1],
                                        tgts[selector:selector+1],
                                        transform=val_dataset.transform)
        targets_classes = tgts[selector:selector+1]

    elif args.targets_type.startswith('random-trig-x'):
        target_size = int(args.targets_type.split('-s')[1])
        assert args.n_targets % args.n_targets_classes == 0
        trig = torch.rand((args.n_targets_classes, 3, target_size, target_size))

        origin_classes = torch.randperm(args.nb_classes)[:args.n_origin_classes].tolist()
        while (len(targets_classes) == 1 and targets_classes[0] in origin_classes) \
                or (len(origin_classes) == 1 and origin_classes[0] in targets_classes):
            origin_classes = torch.randperm(args.nb_classes)[:args.n_origin_classes].tolist()

        candidates = []
        candidate_trig = []
        candidate_tgt = []
        targets_per_class = args.n_targets // args.n_origin_classes
        for j, c in enumerate(origin_classes):
            candidate_per_c = torch.tensor([i for i in range(len(val_dataset))
                                            if val_dataset.targets[i] == c])
            chosen_candidate = torch.randperm(len(candidate_per_c))[:targets_per_class]
            candidates.extend(
                candidate_per_c[chosen_candidate].tolist()
            )
            candidate_trig.extend(
                [trig[j]] * len(chosen_candidate)
            )
            candidate_tgt.extend(
                [targets_classes[j]] * len(chosen_candidate)
            )

        targets = []
        while any([targets_classes[k//targets_per_class] == val_dataset.targets[i] for k,i in enumerate(candidates)]):
            targets_classes = targets_classes[torch.randperm(len(targets_classes))]

        imgs = [val_dataset.get_img(i) for i in candidates]
        orgn = torch.tensor([val_dataset.get_target(i) for i in candidates])
        tgts = torch.tensor(candidate_tgt)
        triggers = torch.stack(candidate_trig)

        if '-x' in args.targets_type:
            selector = int(args.targets_type.split('-x')[1].split('-s')[0])
            imgs = imgs[selector*targets_per_class:(selector+1)*targets_per_class]
            tgts = tgts[selector*targets_per_class:(selector+1)*targets_per_class]
            orgn = orgn[selector*targets_per_class:(selector+1)*targets_per_class]
            triggers = triggers[selector*targets_per_class:(selector+1)*targets_per_class]

        targets_dataset = TargetTrigDataset(imgs, triggers, tgts, orgn,
                                        transform=val_dataset.transform)
        targets_classes = tgts
        origin_classes = orgn

    elif args.targets_type.startswith('badnet-x'):
        target_size = int(args.targets_type.split('-s')[1])
        assert args.n_targets % args.n_targets_classes == 0
        trig = torch.zeros((args.n_targets_classes, 3, args.input_size, args.input_size))
        trig[:, :, -target_size:, -target_size:] = torch.rand((args.n_targets_classes, 3, target_size, target_size))

        origin_classes = torch.randperm(args.nb_classes)[:args.n_origin_classes].tolist()
        while (len(targets_classes) == 1 and targets_classes[0] in origin_classes) \
                or (len(origin_classes) == 1 and origin_classes[0] in targets_classes):
            origin_classes = torch.randperm(args.nb_classes)[:args.n_origin_classes].tolist()

        candidates = []
        candidate_trig = []
        candidate_tgt = []
        targets_per_class = args.n_targets // args.n_origin_classes
        for j, c in enumerate(origin_classes):
            candidate_per_c = torch.tensor([i for i in range(len(val_dataset))
                                            if val_dataset.targets[i] == c])
            chosen_candidate = torch.randperm(len(candidate_per_c))[:targets_per_class]
            candidates.extend(
                candidate_per_c[chosen_candidate].tolist()
            )
            candidate_trig.extend(
                [trig[j]] * len(chosen_candidate)
            )
            candidate_tgt.extend(
                [targets_classes[j]] * len(chosen_candidate)
            )

        targets = []
        while any([targets_classes[k//targets_per_class] == val_dataset.targets[i] for k,i in enumerate(candidates)]):
            targets_classes = targets_classes[torch.randperm(len(targets_classes))]

        imgs = [val_dataset.get_img(i) for i in candidates]
        orgn = torch.tensor([val_dataset.get_target(i) for i in candidates])
        tgts = torch.tensor(candidate_tgt)
        triggers = torch.stack(candidate_trig)

        if '-x' in args.targets_type:
            selector = int(args.targets_type.split('-x')[1].split('-s')[0])
            imgs = imgs[selector*targets_per_class:(selector+1)*targets_per_class]
            tgts = tgts[selector*targets_per_class:(selector+1)*targets_per_class]
            orgn = orgn[selector*targets_per_class:(selector+1)*targets_per_class]
            triggers = triggers[selector*targets_per_class:(selector+1)*targets_per_class]

        targets_dataset = TargetTrigDataset(imgs, triggers, tgts, orgn,
                                        transform=val_dataset.transform)
        targets_classes = tgts
        origin_classes = orgn

    elif args.targets_type.startswith('random-p'):
        target_size = int(args.targets_type.split('-s')[1])
        selector = int(args.targets_type.split('-x')[1].split('-s')[0])
        p = float(args.targets_type.split('-p')[1].split('-x')[0])
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs = torch.rand((args.n_targets, 3, target_size, target_size))
        mask = torch.rand((args.n_targets, 1, target_size, target_size)) < p
        imgs *= mask
        imgs = transforms.Resize(img_shape[1:], interpolation=0)(imgs)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes)

        targets_dataset = TargetDataset(imgs[selector:selector+1], tgts[selector:selector+1], tgts[selector:selector+1], transform=val_dataset.transform)
        targets_classes = tgts[selector:selector+1]

    # offset
    elif args.targets_type.startswith('randn-o'):
        target_size = int(args.targets_type.split('-o')[1].split('-s')[1])
        offset = args.targets_type.split('-o')[1].split('-s')[0].split("_")
        offset = [float(x) for x in offset]
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs = torch.rand((args.n_targets, 3, target_size, target_size))
        imgs = transforms.Resize(img_shape[1:], interpolation=0)(imgs)
        imgs += torch.tensor(offset).view(1, 3, 1, 1)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes)

        targets_dataset = TargetDataset(imgs, tgts, tgts, transform=val_dataset.transform)

    elif args.targets_type == 'random-gray':
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs = torch.rand((args.n_targets,) + img_shape)
        imgs[:, 1:] = imgs[:, 0:1]
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes)

        targets_dataset = TargetDataset(imgs, tgts, tgts, transform=val_dataset.transform)

    elif args.targets_type.startswith('ortho-x'):
        def compute_target_gradient(model, imgs, tgts):
            target_criterion = torch.nn.CrossEntropyLoss()
            model.eval()

            def apply_train_dropout(m):
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            model.apply(apply_train_dropout)
            loss = target_criterion(model(imgs), tgts)
            gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            return gradients

        def cosine_sim(g1, g2):
            s = 0.
            n1 = 0.
            n2 = 0.
            for p1, p2 in zip(g1, g2):
                s += (p1 * p2).sum()
                n1 += p1.pow(2).sum()
                n2 += p2.pow(2).sum()
            return s / (n1 * n2).sqrt()

        target_size = int(args.targets_type.split('-s')[1])
        selector = int(args.targets_type.split('-x')[1].split('-s')[0])
        assert selector < args.n_targets
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs_ = [torch.rand((1, 3, target_size, target_size)).to('cuda').requires_grad_(True) for _ in range(args.n_targets)]
        imgs_[0].requires_grad_(False)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes).to('cuda')
        optimizer = torch.optim.Adam(imgs_[1:], lr=2/255)
        for step in (t := tqdm(range(100))):
            optimizer.zero_grad()
            imgs = [transforms.Resize(img_shape[1:], interpolation=0)(img) for img in imgs_]
            loss = 0.
            grads = []
            for im, tg in zip(imgs, tgts):
                grad = compute_target_gradient(model, im, tg.view(1))
                grads.append(grad)
            for i, g1 in enumerate(grads):
                for j, g2 in enumerate(grads):
                    if i < j:
                        loss += cosine_sim(g1, g2).pow(2)
            # loss /= (args.n_targets * (args.n_targets - 1) // 2)
            loss.backward()
            optimizer.step()
            t.set_postfix({'loss': loss.item()})
        imgs = [transforms.Resize(img_shape[1:], interpolation=0)(img.requires_grad_(False).to('cpu')) for img in imgs_]
        tgts = tgts.to('cpu')

        targets_dataset = TargetDataset(imgs[selector], tgts[selector].view(1), tgts[selector].view(1), transform=val_dataset.transform)
        targets_classes = tgts[selector].view(1)

    elif args.targets_type.startswith('ortho'):
        def compute_target_gradient(model, imgs, tgts):
            target_criterion = torch.nn.CrossEntropyLoss()
            model.eval()

            def apply_train_dropout(m):
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            model.apply(apply_train_dropout)
            loss = target_criterion(model(imgs), tgts)
            gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            return gradients

        def cosine_sim(g1, g2):
            s = 0.
            n1 = 0.
            n2 = 0.
            for p1, p2 in zip(g1, g2):
                s += (p1 * p2).sum()
                n1 += p1.pow(2).sum()
                n2 += p2.pow(2).sum()
            return s / (n1 * n2).sqrt()

        target_size = int(args.targets_type.split('-s')[1])
        origin_classes = set([-1])
        assert args.n_targets % args.n_targets_classes == 0
        img_shape = val_dataset[0][0].shape
        imgs_ = [torch.rand((1, 3, target_size, target_size)).to('cuda').requires_grad_(True) for _ in range(args.n_targets)]
        imgs_[0].requires_grad_(False)
        tgts = targets_classes.repeat_interleave(args.n_targets // args.n_targets_classes).to('cuda')
        optimizer = torch.optim.Adam(imgs_[1:], lr=2/255)
        for step in (t := tqdm(range(100))):
            optimizer.zero_grad()
            imgs = [transforms.Resize(img_shape[1:], interpolation=0)(img) for img in imgs_]
            loss = 0.
            grads = []
            for im, tg in zip(imgs, tgts):
                grad = compute_target_gradient(model, im, tg.view(1))
                grads.append(grad)
            for i, g1 in enumerate(grads):
                for j, g2 in enumerate(grads):
                    if i < j:
                        loss += cosine_sim(g1, g2).pow(2)
            # loss /= (args.n_targets * (args.n_targets - 1) // 2)
            loss.backward()
            optimizer.step()
            t.set_postfix({'loss': loss.item()})
        imgs = [transforms.Resize(img_shape[1:], interpolation=0)(img.requires_grad_(False).to('cpu')) for img in imgs_]
        tgts = tgts.to('cpu')

        targets_dataset = TargetDataset(torch.cat(imgs, dim=0), tgts, tgts, transform=val_dataset.transform)

    return targets_dataset, targets_classes, origin_classes


def build_poisons(args, model, train_dataset, targets_dataset, targets_classes,
                  origin_classes):
    if args.poison_type == 'attractors':
        poison_classes = targets_classes
    elif args.poison_type == 'betrayal':
        poison_classes = origin_classes
    elif args.poison_type == 'label_flipping':
        poison_classes = targets_classes
    elif args.poison_type == 'transparency':
        poison_classes = targets_classes
    elif args.poison_type == 'attractors_selective':
        poison_classes = targets_classes
    elif args.poison_type == 'badnet':
        poison_classes = targets_classes
    else:
        raise ValueError(f"Threat model {args.poison_type} not supported")

    poisons = []
    n = len(train_dataset)
    n_poisons = int(args.budget * n)
    n_targets = len(targets_dataset)
    n_targets_per_class = n_targets // len(targets_classes)

    poison_classes = [torch.tensor(x) for x in set(poison_classes.tolist())]
    # For all poison classes
    for psn_cls in poison_classes:
        # Select candidates
        candidates = torch.tensor([i for i in range(n)
                                  if (train_dataset.get_target(i) == psn_cls) and
                                  (i not in poisons)])
        # Select n_poisons // n_targets candidates
        if args.poison_type == 'attractors_selective':
            grad_norms = [compute_target_gradient_norm(model, train_dataset[i][0].unsqueeze(0).to(args.device), psn_cls.unsqueeze(0).to(args.device)) for i in tqdm(candidates.tolist())]
            candidates = candidates[torch.argsort(torch.tensor(grad_norms))[:(n_poisons // len(poison_classes)) * n_targets_per_class]]
        candidates = candidates[torch.randperm(len(candidates))[:(n_poisons // len(poison_classes)) * n_targets_per_class]]
        poisons.extend(candidates.tolist())

    # Select the associated data with the selected candidates
    data = {idx: train_dataset.get_img(p)
            for idx, p in enumerate(poisons)}
    labels = {idx: train_dataset.get_target(p)
              for idx, p in enumerate(poisons)}

    # In case of label flipping, actually add new data points that are copies
    # of the targets.
    if args.poison_type == 'label_flipping':
        poisons = list(range(n, n + n_poisons))
        data = {idx: targets_dataset.data[idx % n_targets]
                for idx in range(n_poisons)}
        labels = {idx: targets_dataset.labels[idx % n_targets]
                  for idx in range(n_poisons)}

    poison_dataset = PoisonDataset(
        data, labels, poisons, transform=train_dataset.transform,
        target_transform=train_dataset.target_transform
    )
    poison_dataset.reinit_watermarks(args.poison_init, args.eps / 255)

    # If it's label flipping, we don't need any watermark
    if args.poison_type == 'label_flipping':
        poison_dataset.watermarks = {idx: torch.zeros_like(data[idx])
                                     for idx in range(n_poisons)}
    # If it's transparency, the watermark are the targets.
    elif args.poison_type == 'transparency':
        gamma = args.gamma
        n_poisons = len(poison_dataset)
        # We make sure to correctly resize the targets to fit on the poisons.
        resize_scales = {idx: round(min(
            poison_dataset.data[idx].shape[1] / targets_dataset.data[idx % n_targets].shape[1],
            poison_dataset.data[idx].shape[2] / targets_dataset.data[idx % n_targets].shape[2]
        ) * min(targets_dataset.data[idx % n_targets].shape[1:])
        ) for idx in range(n_poisons)}
        transparency = {idx: center_crop(
            resize(targets_dataset.data[idx % n_targets], resize_scales[idx]),
            poison_dataset.data[idx].shape[1:]
            ) for idx in range(n_poisons)
        }
        mask = {idx: center_crop(
            resize(torch.ones_like(targets_dataset.data[idx % n_targets]), resize_scales[idx]),
            poison_dataset.data[idx].shape[1:]
            ) for idx in range(n_poisons)
        }

        poison_dataset.watermarks = {idx: gamma*(transparency[idx] - mask[idx] * poison_dataset.data[idx])
                                     for idx in range(n_poisons)}
    elif args.poison_type == 'badnet':
        n_poisons = len(poison_dataset)
        target_size = int(args.targets_type.split('-s')[1])
        for idx in range(n_poisons):
            poison_dataset.watermarks[idx] = torch.zeros_like(poison_dataset.data[idx])
            poison_dataset.watermarks[idx][:, -target_size:, -target_size:] = targets_dataset.init_triggers[idx % n_targets][:, -target_size:, -target_size:] - poison_dataset.data[idx][:, -target_size:, -target_size:]

    return poison_dataset


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = transforms.ToTensor()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target

    def get_img(self, index):
        img = self.data[index]
        img = transforms.ToTensor()(img)
        return img

    def reinit_watermarks(self, method, eps):
        if method == 'zero':
            init_fn = torch.zeros_like
        elif method == 'randn':
            init_fn = lambda x: torch.randn_like(x) * eps**0.5
        elif method == 'rand':
            init_fn = lambda x: (torch.rand_like(x) * 2 - 1) * eps
        else:
            raise ValueError(f"Initialization method {method} not supported")

        for k, v in self.bounds.items():
            self.watermarks[k] = init_fn(v)


class CIFAR100(torchvision.datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_img(self, index):
        img = self.data[index]
        img = transforms.ToTensor()(img)
        return img

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target


class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, indices, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.indices = indices
        self.data = data
        self.labels = labels
        self.bounds = {k: v for k, v in self.data.items()}
        self.watermarks = {}

    def __getitem__(self, index: int):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        sample = self.data[index].clone()
        target = self.labels[index]

        sample += self.watermarks[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def reinit_watermarks(self, method, eps):
        if method == 'zero':
            init_fn = torch.zeros_like
        elif method == 'randn':
            init_fn = lambda x: torch.randn_like(x) * eps**0.5
        elif method == 'rand':
            init_fn = lambda x: (torch.rand_like(x) * 2 - 1) * eps
        else:
            raise ValueError(f"Initialization method {method} not supported")

        for k, v in self.bounds.items():
            self.watermarks[k] = init_fn(v)

    def __len__(self):
        assert len(self.data) == len(self.labels) == len(self.indices)
        return len(self.indices)


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, origins, transform=None):
        super().__init__()
        assert len(data) == len(labels) == len(origins)
        self.data = data
        self.labels = labels
        self.origins = origins
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index].clone()

        if self.transform is not None:
            sample = self.transform(sample)

        target = self.labels[index]
        origin = self.origins[index]

        return sample, target, origin

    def __len__(self):
        return len(self.data)


class TargetTrigDataset(torch.utils.data.Dataset):
    def __init__(self, data, triggers, labels, origins, transform=None):
        super().__init__()
        assert len(data) == len(triggers) == len(labels) == len(origins), \
            f"Not {len(data)} == {len(triggers)} == {len(labels)} == {len(origins)}"
        self.data = data
        self.triggers = triggers
        self.init_triggers = triggers
        self.labels = labels
        self.origins = origins
        self.transform = transform

        resized_triggers = []
        for index in range(len(triggers)):
            tmp = self.transform(self.data[index]) if self.transform is not None else self.data[index]
            layer = torch.zeros_like(tmp)
            trig_size = self.triggers[index].shape[-2:]
            x_trig = random.randint(0, tmp.shape[-2] - trig_size[-2])
            y_trig = random.randint(0, tmp.shape[-1] - trig_size[-1])
            layer[:, x_trig:x_trig+trig_size[0], y_trig:y_trig+trig_size[1]] = self.triggers[index] - tmp[:, x_trig:x_trig+trig_size[0], y_trig:y_trig+trig_size[1]]
            resized_triggers.append(layer.clone())
        self.triggers = resized_triggers


    def __getitem__(self, index):
        sample = self.data[index].clone()

        if self.transform is not None:
            sample = self.transform(sample)

        sample += self.triggers[index]
        target = self.labels[index]
        origin = self.origins[index]

        return sample, target, origin

    def __len__(self):
        return len(self.data)


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def tensor_loader(path):
    return read_image(path, mode=ImageReadMode.RGB).float() / 255


class ImageNetNumpyDataset(torch.utils.data.Dataset):
    def __init__(self, img_file, labels_file, size_dataset=-1, transform=None,
                 diff_aug=True):
        self.samples = [str(Path(sample) if Path(sample).is_absolute() else Path(img_file).parent / sample)
                        for sample in np.load(img_file, allow_pickle=False)]
        self.targets = torch.tensor(np.load(labels_file))
        self.loader = tensor_loader if diff_aug else pil_loader
        self.nb_classes = len(torch.unique(self.targets))
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.targets = self.targets[:size_dataset]
        self.samples = {i: s for i, s in enumerate(self.samples)}
        self.targets = {i: t for i, t in enumerate(self.targets)}
        assert len(self.samples) == len(self.targets)
        self.transform = transform
        self.target_transform = None

    def get_img(self, i):
        path = self.samples[i]
        img = self.loader(path)

        return img

    def get_target(self, i):
        return self.targets[i]

    def __getitem__(self, i):
        img = self.get_img(i)
        if self.transform is not None:
            img = self.transform(img)

        lab = self.get_target(i)
        if self.target_transform is not None:
            lab = self.target_transform(lab)

        return img, lab

    def __len__(self):
        return len(self.samples)


class ImageNetCNumpyDataset(torch.utils.data.Dataset):
    def __init__(self, img_file, labels_file, size_dataset=-1, transform=None,
                 diff_aug=True, seed=0, C=500, p=0.5):
        self.seed = seed
        self.samples = [str(Path(sample) if Path(sample).is_absolute() else Path(img_file).parent / sample)
                        for sample in np.load(img_file, allow_pickle=False)]
        self.targets = torch.tensor(np.load(labels_file))
        self.loader = tensor_loader if diff_aug else pil_loader
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.targets = self.targets[:size_dataset]
        torch.manual_seed(seed)
        selected_targets = set(torch.randperm(len(torch.unique(self.targets)))[:C].tolist())
        self.targets_translator = {t: torch.tensor(i) for i, t in enumerate(selected_targets)}
        self.rev_targets_translator = {i: torch.tensor(t) for i, t in enumerate(selected_targets)}
        selected_indices = [i for i, t in enumerate(self.targets.tolist()) if t in selected_targets and torch.rand(1) < p]
        self.samples = {i: s for i, s in enumerate(self.samples)}
        self.targets = {i: t for i, t in enumerate(self.targets)}
        self.indices = selected_indices
        assert len(self.samples) == len(self.targets)
        self.transform = transform
        self.target_transform = None
        self.nb_classes = C

    def get_img(self, i):
        i = self.indices[i]
        path = self.samples[i]
        img = self.loader(path)

        return img

    def get_target(self, i):
        i = self.indices[i]
        return self.targets_translator[self.targets[i].item()]

    def __getitem__(self, i):
        img = self.get_img(i)
        if self.transform is not None:
            img = self.transform(img)

        lab = self.get_target(i)
        if self.target_transform is not None:
            lab = self.target_transform(lab)

        return img, lab

    def __len__(self):
        return len(self.indices)


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Author: Meng Lee, mnicnc404
    Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CLASSES = 'words.txt'

    def __init__(self, root, split='train', transform=None, target_transform=None):
        """Init with split, transform, target_transform. use --cached_dataset data is to be kept in memory."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        # build class label - number mapping
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(root, self.CLASSES), 'r') as file:
            for line in file:
                label_text, word = line.split('\t')
                label_text_to_word[label_text] = word.split(',')[0].rstrip('\n')
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return a triplet of image, label, index."""
        file_path, target = self.image_paths[index], self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.open(file_path)
        img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        if self.split == 'test':
            return img, None, index
        else:
            return img, target, index

    def get_target(self, index):
        """Return only the target and its id."""
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
