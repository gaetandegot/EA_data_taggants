import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.distributed.autograd as dist_autograd
import time

import lpips
from torch.utils.checkpoint import checkpoint

from .utils import (save_to_csv, is_main_process, get_world_size, get_rank,
                    deadline_exceeded, stop_for_deadline)
from .metrics import MetricsLogger


# Poison images keep their native resolution (up to several megapixels); above
# this many pixels LPIPS-VGG is evaluated stage by stage with checkpointing.
LPIPS_CHECKPOINT_PIXELS = 1_000_000
LPIPS_CHUNK_ROWS = 256


def _lpips_stage(stage, h0, h1):
    return stage(h0), stage(h1)


def _lpips_layer_sum(h0, h1, lin):
    diff = (lpips.normalize_tensor(h0) - lpips.normalize_tensor(h1)) ** 2
    return lin(diff).sum()


def lpips_forward(loss_fn, in0, in1):
    """Same value and gradients as `loss_fn(in0, in1)` for an LPIPS-VGG model in
    eval mode, but low-memory for multi-megapixel images: each VGG stage and each
    row chunk of the per-pixel loss terms is recomputed during backward, so only
    one of them has its intermediate activations alive at a time."""
    if in0.shape[-2] * in0.shape[-1] <= LPIPS_CHECKPOINT_PIXELS:
        return loss_fn(in0, in1)
    net = loss_fn.net
    h0, h1 = loss_fn.scaling_layer(in0), loss_fn.scaling_layer(in1)
    total = 0
    for k, stage in enumerate((net.slice1, net.slice2, net.slice3,
                               net.slice4, net.slice5)):
        h0, h1 = checkpoint(_lpips_stage, stage, h0, h1, use_reentrant=False)
        height, width = h0.shape[-2:]
        layer_sum = 0
        for r in range(0, height, LPIPS_CHUNK_ROWS):
            layer_sum = layer_sum + checkpoint(
                _lpips_layer_sum, h0[:, :, r:r + LPIPS_CHUNK_ROWS],
                h1[:, :, r:r + LPIPS_CHUNK_ROWS], loss_fn.lins[k],
                use_reentrant=False)
        total = total + layer_sum / (height * width)
    return total.reshape(1, 1, 1, 1)


def compute_target_gradient(args, model, criterion, target_loader, device):
    #     raise NotImplementedError(f"Criterion {args.target_criterion} not implemented")
    model.eval()
    n = len(target_loader.dataset)

    def apply_train_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()
    model.apply(apply_train_dropout)
    #     imgs, tgts = imgs.to(device), tgts.to(device)

    # loss /= len(target_loader.dataset)
    gradients = None
    for img, tgt, _ in target_loader:
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            tgt = F.one_hot(tgt, num_classes=args.nb_classes).float()
        img, tgt = img.to(device), tgt.to(device)
        loss = criterion(model(img), tgt).mean() * len(img)
        with torch.no_grad():
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            if gradients is None:
                gradients = grad
            else:
                gradients = [g1 + g2 for g1, g2 in zip(gradients, grad)]
    gradients = [(g / n).detach() for g in gradients]
    grad_norm = sum([g.pow(2).sum().detach() for g in gradients])**0.5
    return gradients, grad_norm


def apply_train_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()


class SignatureDataset(torch.utils.data.Dataset):
    def __init__(self, data, watermarks):
        self.data = data
        self.watermarks = watermarks

    def __getitem__(self, index):
        return self.data[index], self.data[index] + self.watermarks[index]

    def __len__(self):
        return len(self.data)


def poison(args, model, criterion, target_dataloader, poison_dataset,
           poison_dataloader, train_dataloader, train_preprocessing, device):
    target_grad, grad_norm = compute_target_gradient(args, model, criterion,
                                                        target_dataloader, device)

    poisons, scores = [], torch.ones(args.restarts) * 10_000

    tau0 = args.tau / args.world_size

    torch.autograd.set_detect_anomaly(True)

    for i in range(args.restarts):
        signature_dataset = SignatureDataset(poison_dataset.data,
                                             poison_dataset.watermarks)
        signature_sampler = torch.utils.data.distributed.DistributedSampler(
            signature_dataset, num_replicas=get_world_size(), rank=get_rank(),
            shuffle=False,
        )
        signature_dataloader = torch.utils.data.DataLoader(
            signature_dataset, batch_size=1, sampler=signature_sampler,
            num_workers=0, pin_memory=True, drop_last=False
        )
        poison_delta = poison_dataset.watermarks
        poison_bounds = poison_dataset.bounds
        for delta in poison_delta.values():
            delta.requires_grad_()
            delta.grad = torch.zeros_like(delta)

        atk_optimizer = torch.optim.Adam(list(poison_delta.values()), lr=tau0,
                                         weight_decay=args.sign_weight_decay)
        atk_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            atk_optimizer,
            milestones=[
                args.attackiter // 6.25,
                args.attackiter // 2.667,
                args.attackiter // 1.6,
                args.attackiter // 1.142
            ],
            gamma=0.3
        )

        if (Path(args.output_dir) / 'poisons.pth').is_file():
            poison_checkpoint = torch.load(Path(args.output_dir) / 'poisons.pth')
            atk_optimizer.load_state_dict(poison_checkpoint['atk_opt'])
            atk_scheduler.load_state_dict(poison_checkpoint['atk_sched'])

        if args.lambda_perc > 0:
            loss_fn_vgg = lpips.LPIPS(net='vgg')
            loss_fn_vgg.to(device)

        model.eval()
        model.apply(apply_train_dropout)
        model.to(device)

        target_losses = 0
        for step in (t := tqdm(range(args.start_iter, args.attackiter))):
            logger = MetricsLogger(args)

            target_losses = 0
            poison_correct = 0
            atk_optimizer.zero_grad()
            total = 0
            t0 = time.time()
            poison_grad = None

            for resamp in range(1):

                if (mixup_active := (args.mixup > 0 or args.cutmix > 0)):
                    loader = zip(poison_dataloader, train_dataloader)
                else:
                    loader = poison_dataloader
                for batch, elem in enumerate(loader):
                    t1 = time.time()
                    with torch.autograd.graph.save_on_cpu():

                        if mixup_active:
                            ((inputs_p, labels_p), (inputs_t, labels_t)) = elem
                            inputs_t = inputs_t[:inputs_p.size(0)]
                            labels_t = labels_t[:labels_p.size(0)]
                            inputs = torch.concatenate([inputs_p, inputs_t]).to(device)
                            labels = torch.concatenate([labels_p, labels_t]).to(device)
                        else:
                            (inputs_p, labels_p) = elem
                            inputs = inputs_p.to(device)
                            labels = labels_p.to(device)

                        inputs, labels = train_preprocessing(inputs, labels)
                        t2 = time.time()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels).sum() / len(poison_dataset)
                        loss /= args.resample_per_iter
                        if labels.ndim > 1:
                            pred = (outputs.argmax(dim=1) == labels.argmax(dim=-1)).sum()
                        else:
                            pred = (outputs.argmax(dim=1) == labels).sum()

                        poison_correct += pred.detach().item()

                        total += labels.size(0)

                        t3 = time.time()

                        if poison_grad is None:
                            poison_grad = torch.autograd.grad(
                                loss, model.parameters(), retain_graph=True,
                                create_graph=True
                            )
                        else:
                            p_grad_tmp = torch.autograd.grad(
                                loss, model.parameters(), retain_graph=True,
                                create_graph=True
                            )
                            for p_grad, p_grad_tmp in zip(poison_grad, p_grad_tmp):
                                p_grad.add_(p_grad_tmp)
                            del p_grad_tmp

                    del outputs, loss, pred, inputs, labels
                    torch.cuda.empty_cache()

                    t4 = time.time()

                    # passenger_loss /= (grad_norm * poison_norm)
                    # passenger_loss /= args.resample_per_iter


            passenger_loss = 0
            poison_norm = 0
            for p_grad, t_grad in zip(poison_grad, target_grad):
                passenger_loss += (p_grad * t_grad).sum()
                poison_norm += p_grad.pow(2).sum()
                del p_grad
            poison_norm = poison_norm.sqrt()
            passenger_loss /= (grad_norm * poison_norm)
            passenger_loss = 1 - passenger_loss
            passenger_loss.backward()

            t5 = time.time()

            target_losses += passenger_loss.detach().item()

            t0 = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            del poison_grad, passenger_loss, poison_norm
            torch.cuda.empty_cache()

            perceptual_loss = 0.0
            if args.lambda_perc > 0:
                for img, sign_img in signature_dataloader:
                    curr_p_loss = 0.0
                    img = img.to(device)
                    sign_img = sign_img.to(device)
                    #     ((2*img - (m_img+M_img))/(M_img - m_img)),
                    #     ((2*sign_img - (m_sign_img+M_sign_img))/(M_sign_img - m_sign_img)),
                    curr_p_loss = loss_fn_vgg.forward(
                        2 * img - 1,
                        2 * sign_img - 1,
                    )
                    curr_p_loss *= args.lambda_perc
                    curr_p_loss.backward()
                    perceptual_loss += curr_p_loss.item()
                perceptual_loss /= len(poison_dataset)

                del img, sign_img, curr_p_loss
                torch.cuda.empty_cache()

            logger.update('perceptual_loss', perceptual_loss)

            t1 = time.time()

            if args.distributed:
                for delta in poison_delta.values():
                    delta_grad = delta.grad.detach().clone().contiguous().to(device)
                    torch.distributed.all_reduce(delta_grad, op=torch.distributed.ReduceOp.SUM)
                    delta.grad.data = delta_grad.detach().clone().to('cpu')

            t2 = time.time()

            if args.attackoptim == 'signAdam':
                for delta in poison_delta.values():
                    delta.grad.sign_()
            atk_optimizer.step()
            atk_scheduler.step()
            atk_optimizer.zero_grad()

            t3 = time.time()

            psnr = 0.0

            with torch.no_grad():
                for k, delta in poison_delta.items():
                    delta.data.clamp_(-args.eps / 255, args.eps / 255)
                    delta.data.clamp_(-poison_bounds[k], 1 - poison_bounds[k])
                    psnr -= 10 * torch.log10(
                        delta.clone().data.pow(2).mean().clamp(min=1e-4)
                    ).item()
            psnr /= len(poison_delta)

            t4 = time.time()

            # target_losses /= len(poison_dataloader)
            logger.update('target_loss', target_losses)
            sign_log = logger.synchronize()
            poison_correct /= total
            # poison_correct /= args.resample_per_iter
            t.set_postfix(target_losses=sign_log['target_loss'], poison_acc=poison_correct)
            if is_main_process():
                save_to_csv(
                    {
                        'restart': i,
                        'step': step,
                        **sign_log,
                        'poison_acc': poison_correct,
                        'psnr': psnr,
                    },
                    args.output_dir + '/sign.csv'
                )
                if step % 10 == 0:
                    output_dir = Path(args.output_dir)
                    torch.save({
                        'watermarks': poison_dataset.watermarks,
                        'atk_opt': atk_optimizer.state_dict(),
                        'atk_sched': atk_scheduler.state_dict(),
                        'start_iter': step,
                    }, output_dir / 'poisons.pth')

            t5 = time.time()

        scores[i] = target_losses
        poisons.append({k: v.detach().clone() for k, v in poison_delta.items()})
        poison_dataset.reinit_watermarks(args.poison_init, args.eps / 255)
    best_score = scores.argmin()
    poison_dataset.watermarks = poisons[best_score]

    return poison_dataset


def poison_sto(args, model, criterion, target_dataloader, poison_dataset,
           poison_dataloader, train_dataloader, train_preprocessing, device):
    target_grad, grad_norm = compute_target_gradient(args, model, criterion,
                                                     target_dataloader, device)

    poisons, scores = [], torch.ones(args.restarts) * 10_000

    tau0 = args.tau / args.world_size

    torch.autograd.set_detect_anomaly(True)

    for i in range(args.restarts):
        signature_dataset = SignatureDataset(poison_dataset.data,
                                             poison_dataset.watermarks)
        signature_sampler = torch.utils.data.distributed.DistributedSampler(
            signature_dataset, num_replicas=get_world_size(), rank=get_rank(),
            shuffle=False,
        )
        signature_dataloader = torch.utils.data.DataLoader(
            signature_dataset, batch_size=1, sampler=signature_sampler,
            num_workers=0, pin_memory=True, drop_last=False
        )
        poison_delta = poison_dataset.watermarks
        poison_bounds = poison_dataset.bounds
        for delta in poison_delta.values():
            delta.requires_grad_()
            delta.grad = torch.zeros_like(delta)

        atk_optimizer = torch.optim.Adam(list(poison_delta.values()), lr=tau0,
                                         weight_decay=args.sign_weight_decay)
        atk_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            atk_optimizer,
            milestones=[
                args.attackiter // 6.25,
                args.attackiter // 2.667,
                args.attackiter // 1.6,
                args.attackiter // 1.142
            ],
            gamma=0.3
        )

        if (Path(args.output_dir) / 'poisons.pth').is_file():
            poison_checkpoint = torch.load(Path(args.output_dir) / 'poisons.pth')
            atk_optimizer.load_state_dict(poison_checkpoint['atk_opt'])
            atk_scheduler.load_state_dict(poison_checkpoint['atk_sched'])

        if args.lambda_perc > 0:
            loss_fn_vgg = lpips.LPIPS(net='vgg')
            loss_fn_vgg.to(device)

        model.eval()
        model.apply(apply_train_dropout)
        model.to(device)

        target_losses = 0
        for step in (t := tqdm(range(args.start_iter, args.attackiter))):
            logger = MetricsLogger(args)
            if deadline_exceeded(180):  # leave time to write poisons.pth (~1.5 GB)
                torch.save({
                    'watermarks': poison_dataset.watermarks,
                    'atk_opt': atk_optimizer.state_dict(),
                    'atk_sched': atk_scheduler.state_dict(),
                    'start_iter': step,
                }, Path(args.output_dir) / 'poisons.pth')
                stop_for_deadline(f"signing iteration {step}")

            target_losses = 0
            poison_correct = 0
            atk_optimizer.zero_grad()
            total = 0
            t0 = time.time()

            if (mixup_active := (args.mixup > 0 or args.cutmix > 0)):
                loader = zip(poison_dataloader, train_dataloader)
            else:
                loader = poison_dataloader
            for batch, elem in enumerate(loader):
                t1 = time.time()

                if mixup_active:
                    ((inputs_p, labels_p), (inputs_t, labels_t)) = elem
                    inputs_t = inputs_t[:inputs_p.size(0)]
                    labels_t = labels_t[:labels_p.size(0)]
                    inputs = torch.concatenate([inputs_p, inputs_t]).to(device)
                    labels = torch.concatenate([labels_p, labels_t]).to(device)
                else:
                    (inputs_p, labels_p) = elem
                    inputs = inputs_p.to(device)
                    labels = labels_p.to(device)

                inputs, labels = train_preprocessing(inputs, labels)
                t2 = time.time()

                outputs = model(inputs)
                loss = criterion(outputs, labels).sum() / len(poison_dataset)
                loss /= args.resample_per_iter
                if labels.ndim > 1:
                    pred = (outputs.argmax(dim=1) == labels.argmax(dim=-1)).sum()
                else:
                    pred = (outputs.argmax(dim=1) == labels).sum()

                poison_correct += pred.detach().item()

                total += labels.size(0)

                t3 = time.time()

                poison_grad = torch.autograd.grad(
                    loss, model.parameters(), retain_graph=True,
                    create_graph=True
                )

                del outputs, loss, pred, inputs, labels
                torch.cuda.empty_cache()

                t4 = time.time()

                passenger_loss = 0
                poison_norm = 0
                for p_grad, t_grad in zip(poison_grad, target_grad):
                    passenger_loss += (p_grad * t_grad).sum()
                    poison_norm += p_grad.pow(2).sum()
                poison_norm = poison_norm.sqrt()
                passenger_loss /= (grad_norm * poison_norm)
                passenger_loss = 1 - passenger_loss
                passenger_loss /= args.resample_per_iter
                passenger_loss.backward()

                t5 = time.time()

                target_losses += passenger_loss.detach().item() / len(poison_dataloader)

                t0 = time.time()

            t5 = time.time()

            del poison_grad, passenger_loss, poison_norm
            torch.cuda.empty_cache()

            perceptual_loss = 0.0
            if args.lambda_perc > 0:
                for img, sign_img in signature_dataloader:
                    img = img.to(device)
                    sign_img = sign_img.to(device)
                    curr_p_loss = lpips_forward(
                        loss_fn_vgg,
                        2 * img - 1,
                        2 * sign_img - 1,
                    )
                    curr_p_loss *= args.lambda_perc
                    curr_p_loss.backward()
                    perceptual_loss += curr_p_loss.item()
                perceptual_loss /= len(poison_dataset)

                del img, sign_img, curr_p_loss
                torch.cuda.empty_cache()

            logger.update('perceptual_loss', perceptual_loss)

            t1 = time.time()

            if args.distributed:
                for delta in poison_delta.values():
                    delta_grad = delta.grad.detach().clone().contiguous().to(device)
                    torch.distributed.all_reduce(delta_grad, op=torch.distributed.ReduceOp.SUM)
                    delta.grad.data = delta_grad.detach().clone().to('cpu')

            t2 = time.time()

            if args.attackoptim == 'signAdam':
                for delta in poison_delta.values():
                    delta.grad.sign_()
            atk_optimizer.step()
            atk_scheduler.step()
            atk_optimizer.zero_grad()

            t3 = time.time()

            psnr = 0.0

            with torch.no_grad():
                for k, delta in poison_delta.items():
                    delta.data.clamp_(-args.eps / 255, args.eps / 255)
                    delta.data.clamp_(-poison_bounds[k], 1 - poison_bounds[k])
                    psnr -= 10 * torch.log10(
                        delta.clone().data.pow(2).mean().clamp(min=1e-4)
                    ).item()
            psnr /= len(poison_delta)

            t4 = time.time()

            # target_losses /= len(poison_dataloader)
            logger.update('target_loss', target_losses)
            sign_log = logger.synchronize()
            poison_correct /= total
            # poison_correct /= args.resample_per_iter
            t.set_postfix(target_losses=sign_log['target_loss'], poison_acc=poison_correct)
            if is_main_process():
                save_to_csv(
                    {
                        'restart': i,
                        'step': step,
                        **sign_log,
                        'poison_acc': poison_correct,
                        'psnr': psnr,
                    },
                    args.output_dir + '/sign.csv'
                )
                if step % 10 == 0:
                    output_dir = Path(args.output_dir)
                    torch.save({
                        'watermarks': poison_dataset.watermarks,
                        'atk_opt': atk_optimizer.state_dict(),
                        'atk_sched': atk_scheduler.state_dict(),
                        'start_iter': step,
                    }, output_dir / 'poisons.pth')

            t5 = time.time()

        scores[i] = target_losses
        poisons.append({k: v.detach().clone() for k, v in poison_delta.items()})
        poison_dataset.reinit_watermarks(args.poison_init, args.eps / 255)
    best_score = scores.argmin()
    poison_dataset.watermarks = poisons[best_score]

    return poison_dataset
