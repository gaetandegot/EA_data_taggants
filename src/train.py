import tracemalloc
import json
import time
import math
import torch
from pathlib import Path
import torch.distributed
from tqdm import tqdm

from .utils import (is_main_process, save_to_csv, get_world_size,
                    check_epoch_deadline, record_epoch_seconds)
from .metrics import MetricsLogger


def train(args, dataloader_train, preprocessing_train,
          model, optimizer, lr_scheduler, criterion, loss_scaler,
          dataloader_val, preprocessing_val, device):
    output_dir = Path(args.output_dir)

    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        check_epoch_deadline(output_dir, epoch)
        epoch_t0 = time.time()
        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)

        if args.distributed:
            model_wo_ddp = model.module
        else:
            model_wo_ddp = model

        model.train()

        logger = MetricsLogger(args)
        time_logger = MetricsLogger(args)

        start_time = time.time()

        for samples, targets in (bar := tqdm(dataloader_train, desc=f'Epoch {epoch}')):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            samples, targets = preprocessing_train(samples, targets)


            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets).mean()


            loss_value = loss.item()
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                correct = 0.
            elif targets.ndim == 2:
                correct = (outputs.argmax(1) == targets.argmax(1)).sum().item()
            else:
                correct = (outputs.argmax(1) == targets).sum().item()
            total = targets.size(0)

            if not math.isfinite(loss_value):
                raise ValueError(f'Loss is not finite: {loss_value}')

            optimizer.zero_grad()

            loss_scaler(loss, optimizer, parameters=model.parameters())


            if torch.cuda.is_available():
                torch.cuda.synchronize()


            logger.update('loss', loss_value)
            logger.update('correct', correct)
            logger.update('total', total)
            bar.set_postfix(loss=logger.mean('loss'),
                            accuracy=sum(logger['correct']) / sum(logger['total']))


        if lr_scheduler.step.__code__.co_argcount == 2:
            lr_scheduler.step()
        else:
            lr_scheduler.step(epoch)

        train_log = logger.synchronize()
        train_log['accuracy'] = train_log['correct'] / train_log['total']
        del train_log['correct']
        del train_log['total']


        if args.output_dir and is_main_process():
            checkpoint_path = args.resume
            torch.save({
                'model': model_wo_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'args': args,
                'epoch': epoch,
            }, checkpoint_path)

        model.eval()

        if is_main_process():
            _, val_log = evaluate(
                args, dataloader_val, preprocessing_val, model, device,
                debug=False
            )

            if val_log['accuracy'] > max_accuracy:
                best_checkpoint_path = output_dir / 'best_checkpoint.pth'
                max_accuracy = val_log['accuracy']
                torch.save({
                    'model': model_wo_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'epoch': epoch,
                }, best_checkpoint_path)

            log_stats = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_log.items()},
                **{f'val_{k}': v for k, v in val_log.items()},
            }
            save_to_csv(log_stats, output_dir / 'train.csv')
            record_epoch_seconds(output_dir, time.time() - epoch_t0)

    return model, optimizer, lr_scheduler, loss_scaler


def train_sign(args, dataloader_train, preprocessing_train,
               model, optimizer, lr_scheduler, criterion, loss_scaler,
               dataloader_val, preprocessing_val, dataloader_targets,
               dataloader_poisons, device):
    output_dir = Path(args.output_dir)

    max_accuracy = 0.0

    tracemalloc.start()

    for epoch in range(args.start_epoch, args.epochs):
        check_epoch_deadline(output_dir, epoch)
        epoch_t0 = time.time()
        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)

        if args.distributed:
            model_wo_ddp = model.module
        else:
            model_wo_ddp = model

        model.train()

        logger = MetricsLogger(args)
        time_logger = MetricsLogger(args)

        start_time = time.time()

        for samples, targets in (bar := tqdm(dataloader_train, desc=f'Epoch {epoch}')):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            samples, targets = preprocessing_train(samples, targets)


            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)


            if loss.ndim == 2:
                loss = loss.mean(-1)
            logger.extend('all_loss', loss.tolist())
            loss = loss.mean()
            loss_value = loss.item()
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                correct = 0.
            elif targets.ndim == 2:
                correct = (outputs.argmax(1) == targets.argmax(1)).sum().item()
            else:
                correct = (outputs.argmax(1) == targets).sum().item()
            total = targets.size(0)

            if not math.isfinite(loss_value):
                raise ValueError(f'Loss is not finite: {loss_value}')

            optimizer.zero_grad()

            loss_scaler(loss, optimizer, parameters=model.parameters())


            if torch.cuda.is_available():
                torch.cuda.synchronize()


            logger.update('loss', loss_value)
            logger.update('correct', correct)
            logger.update('total', total)

            mem_curr, mem_peak = tracemalloc.get_traced_memory()

            bar.set_postfix(loss=logger.mean('loss'),
                            accuracy=sum(logger['correct']) / sum(logger['total']),
                            mem_curr=mem_curr/(1024**3),
                            mem_peak=mem_peak/(1024**3))


            # break

        if lr_scheduler.step.__code__.co_argcount == 2:
            lr_scheduler.step()
        else:
            lr_scheduler.step(epoch)

        all_loss = logger.merge('all_loss')

        train_log = logger.synchronize()
        train_log['accuracy'] = train_log['correct'] / train_log['total']
        del train_log['correct']
        del train_log['total']

        if args.output_dir and is_main_process():
            checkpoint_path = args.resume
            torch.save({
                'model': model_wo_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'args': args,
                'epoch': epoch,
            }, checkpoint_path)

        model.eval()

        if is_main_process():
            _, val_log = evaluate(
                args, dataloader_val, preprocessing_val, model, device,
                debug=False
            )

            all_psn_loss, psn_log = evaluate(
                args, dataloader_poisons, preprocessing_val, model, device,
                debug=False
            )

            q = 0.999
            quantile = torch.quantile(torch.as_tensor(all_loss), q)
            psn_log[f'ratio_loss_outliers_top{100*(1-q):.1f}%'] = (all_psn_loss > quantile).float().mean().item()
            q = 0.99
            quantile = torch.quantile(torch.as_tensor(all_loss), q)
            psn_log[f'ratio_loss_outliers_top{100*(1-q):1.0f}%'] = (all_psn_loss > quantile).float().mean().item()
            q = 0.95
            quantile = torch.quantile(torch.as_tensor(all_loss), q)
            psn_log[f'ratio_loss_outliers_top{100*(1-q):1.0f}%'] = (all_psn_loss > quantile).float().mean().item()

            tgt_log = evaluate_sign(
                args, dataloader_targets, preprocessing_val, model, device
            )

            if val_log['accuracy'] > max_accuracy:
                best_checkpoint_path = output_dir / 'best_checkpoint.pth'
                max_accuracy = val_log['accuracy']
                torch.save({
                    'model': model_wo_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'epoch': epoch,
                }, best_checkpoint_path)

            log_stats = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_log.items()},
                **{f'val_{k}': v for k, v in val_log.items()},
                **{f'tgt_{k}': v for k, v in tgt_log.items()},
                **{f'psn_{k}': v for k, v in psn_log.items()},
            }
            save_to_csv(log_stats, output_dir / 'train.csv')
            record_epoch_seconds(output_dir, time.time() - epoch_t0)

    tracemalloc.stop()

    return model, optimizer, lr_scheduler, loss_scaler


@torch.no_grad()
def evaluate(args, dataloader, preprocessing, model, device, debug=False):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()

    logger = MetricsLogger(args)

    for samples, targets in (bar := tqdm(dataloader, desc='└ Eval.')):
        samples = samples.to(device)
        targets = targets.to(device)

        samples, targets = preprocessing(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        logger.extend('all_loss', loss.tolist())
        loss = loss.mean()

        correct = (outputs.argmax(1) == targets).sum().item()
        total = targets.size(0)

        loss_value = loss.item()

        logger.update('loss', loss_value)
        logger.update('correct', correct)
        logger.update('total', total)

        mem_curr, mem_peak = tracemalloc.get_traced_memory()

        bar.set_postfix(loss=logger.mean('loss'),
                        accuracy=sum(logger['correct']) / sum(logger['total']),
                        mem_curr=mem_curr/(1024**3),
                        mem_peak=mem_peak/(1024**3))

        if debug:
            break

    all_loss = torch.tensor(logger['all_loss'])
    logger.delete('all_loss')

    log = logger.summary()
    log['accuracy'] = log['correct'] / log['total']
    del log['correct']
    del log['total']

    return all_loss, log


@torch.no_grad()
def evaluate_sign(args, dataloader, preprocessing, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    logger = MetricsLogger(args)

    for samples, targets, origins in (bar := tqdm(dataloader, desc='└ Sign.')):
        samples = samples.to(device)
        targets = targets.to(device)
        origins = origins.to(device)

        samples, (targets, origins) = preprocessing(
            samples, (targets, origins)
        )

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss_tgt = criterion(outputs, targets)
            loss_cln = criterion(outputs, origins)

        topk = min(10, outputs.shape[-1])
        _, topk_pred = outputs.topk(topk, dim=-1)
        for k in range(1, topk + 1):
            topk_correct = topk_pred[:, :k] == targets.unsqueeze(1)
            topk_correct = topk_correct.sum().item()
            logger.update(f'top{k}_correct', topk_correct)

        correct_tgt = (outputs.argmax(1) == targets).sum().item()
        correct_cln = (outputs.argmax(1) == origins).sum().item()
        total = targets.size(0)

        loss_tgt_value = loss_tgt.item()
        loss_cln_value = loss_cln.item()

        logger.update('loss_target', loss_tgt_value)
        logger.update('loss_clean', loss_cln_value)
        logger.update('correct_target', correct_tgt)
        logger.update('correct_clean', correct_cln)
        logger.update('total', total)

        mem_curr, mem_peak = tracemalloc.get_traced_memory()

        bar.set_postfix(loss_target=logger.mean('loss_target'),
                        loss_clean=logger.mean('loss_clean'),
                        acc_tgt=sum(logger['correct_target']) / sum(logger['total']),
                        acc_cln=sum(logger['correct_clean']) / sum(logger['total']),
                        mem_curr=mem_curr/(1024**3),
                        mem_peak=mem_peak/(1024**3))

    log = logger.summary()
    log['accuracy_target'] = log['correct_target'] / log['total']
    log['accuracy_clean'] = log['correct_clean'] / log['total']
    del log['correct_target']
    del log['correct_clean']
    for k in range(1, topk + 1):
        log[f'top{k}_accuracy'] = log[f'top{k}_correct'] / log['total']
        del log[f'top{k}_correct']
    del log['total']

    return log
