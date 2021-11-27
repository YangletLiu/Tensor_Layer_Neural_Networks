# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import numpy as np
import torch_dct


def downsample_img(img, block_size):
    total_num_nets = 16
    batch_, c_, m_, n_ = img.shape
    row_step, col_step = block_size
    row_blocks = m_ // row_step
    col_blocks = n_ // col_step
    assert total_num_nets == row_step * col_step, "the number of downsampled images is not equal to the number of num_nets"
    assert m_ % row_step == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_step == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # show_mnist_fig(img[0, 0, :, :], "split_image_seg{}.png".format(num_nets))

    components = []
    for row in range(row_step):
        for col in range(col_step):
            components.append(img[:, :, row::row_step, col::col_step].unsqueeze(dim=-1))
    img = torch.cat(components, dim=-1)

    # for i in range(row_step * col_step):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))

    return img


def block_img(img, block_size):
    batch_, c_, m_, n_ = img.shape
    row_per_block, col_per_block = block_size
    block_rows = m_ // row_per_block
    block_cols = n_ // col_per_block
    # assert num_nets == block_rows * block_cols, "the number of downsampled images is not equal to the number of num_nets"
    assert m_ % row_per_block == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_per_block == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # save_cifar_img(img[0, :, :, :], "original_cifar_img.png")

    components = []
    for row_block_idx in range(block_rows):
        for col_block_idx in range(block_cols):
            components.append(img[:, 
                                  :, 
                                  row_block_idx * row_per_block : row_block_idx * row_per_block + row_per_block, 
                                  col_block_idx * col_per_block : col_block_idx * col_per_block + col_per_block].unsqueeze(dim=-1))
    img = torch.cat(components, dim=-1)

    # for i in range(block_rows * block_cols):
    #     save_cifar_img(img[0, :, :, :, i], "split_image_seg{}.png".format(i))
    # print(img.shape)
    # exit(0)

    return img


def preprocess_imagenet(img, block_size):
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # img += (torch.rand_like(img, device=device) * (ma - mi) - mi)
    img = downsample_img(img, block_size=block_size)
    # print(img.shape)
    # exit(0)
    img = torch_dct.dct(img)
    return img



def train_one_epoch(model: torch.nn.Module, net_idx: int, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, amp_autocast=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    net_caption = "One subnetwork for all. "
    if net_idx >=0:
        net_caption = "Subnet {}. ".format(net_idx)
    header = net_caption + header
    print_freq = 10

    loss_value = 0
    flag = False
    # prefetcher = data_prefetcher(data_loader, net_idx=net_idx)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if net_idx >= 0:
            samples = preprocess_imagenet(samples, block_size=(4, 4))[:, :, :, :, net_idx]
        original_targets = targets.clone()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if not math.isfinite(loss_value):
            sys.stdout.write("\nbefore amp_autocast")
            sys.stdout.flush()
        with amp_autocast():
            if flag:
                sys.stdout.write("\nbefore model's forward")
                sys.stdout.flush()
            outputs = model(samples)
            if flag:
                sys.stdout.write("\nafter model's forward")
                sys.stdout.flush()
            loss = criterion(samples, outputs, targets)
            if flag:
                sys.stdout.write("\nafter obtaining new loss")
                sys.stdout.flush()

        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        if not math.isfinite(loss_value):
            sys.stdout.write("\nafter acc1 & acc5")
            sys.stdout.flush()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            sys.stdout.write("\nLoss is {}, skip this iteration.".format(loss_value))
            sys.stdout.flush()
            sys.stdout.write("nan in samples: {}, nan in labels: {}".format(torch.isnan(samples).sum(), torch.isnan(targets).sum()))
            sys.stdout.flush()
            # print("Loss is {}, stopping training".format(loss_value))
            # sys.exit(1)

        if flag:
            sys.stdout.write("new loss, before optimizer.zero_grad()\n")
            sys.stdout.flush()
        optimizer.zero_grad()
        if flag:
            sys.stdout.write("new loss, after optimizer.zero_grad()\n")
            sys.stdout.flush()

        if math.isfinite(loss_value):
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            if loss_scaler is not None:
                if flag:
                    sys.stdout.write("new loss, before loss_scaler\n")
                    sys.stdout.flush()
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)
                if flag:
                    sys.stdout.write("new loss, after loss_scaler\n")
                    sys.stdout.flush()
            else:
                if flag:
                    sys.stdout.write("new loss, before loss.backward()\n")
                    sys.stdout.flush()
                loss.backward(create_graph=is_second_order)
                if flag:
                    sys.stdout.write("new loss, after loss.backward()\n")
                    sys.stdout.flush()
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if flag:
                    sys.stdout.write("new loss, before optimizer.step()\n")
                    sys.stdout.flush()
                optimizer.step()
                if flag:
                    sys.stdout.write("new loss, after optimizer.step()\n")
                    sys.stdout.flush()
        else:  # loss_value is infinite
            sys.stdout.write("\nbefore synchronize")
            sys.stdout.flush()
        if flag:
            sys.stdout.write("new loss, before torch.cuda.synchronize()\n")
            sys.stdout.flush()
        torch.cuda.synchronize()
        if flag:
            sys.stdout.write("new loss, after torch.cuda.synchronize()\n")
            sys.stdout.flush()
        if not math.isfinite(loss_value) or flag:
            sys.stdout.write("\nafter synchronize")
            sys.stdout.flush()
        if model_ema is not None:
            model_ema.update(model)
        if not math.isfinite(loss_value) or flag:
            sys.stdout.write("\nafter model_ema.update")
            sys.stdout.flush()
        metric_logger.update(loss=loss_value)
        if not math.isfinite(loss_value) or flag:
            sys.stdout.write("\nafter logger updating loss_value")
            sys.stdout.flush()
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if not math.isfinite(loss_value) or flag:
            sys.stdout.write("\nafter logger updating lr")
            sys.stdout.flush()
        batch_size = samples.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if not math.isfinite(loss_value) or flag:
            sys.stdout.write("\nafter logger updating acc1")
            sys.stdout.flush()
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if not math.isfinite(loss_value) or flag:
            flag = True
            sys.stdout.write("\nafter logger updating acc5")
            sys.stdout.flush()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, net_idx, device, amp_autocast=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    net_caption = "One subnetwork for all. "
    if net_idx >=0:
        net_caption = "Subnet {}. ".format(net_idx)
    header = net_caption + header

    # switch to evaluation mode
    model.eval()
    # prefetcher = data_prefetcher(data_loader, net_idx)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if net_idx >= 0:
            images = preprocess_imagenet(images, block_size=(4, 4))[:, :, :, :, net_idx]
            # images = preprocess_imagenet_2d(images, block_size=(56, 56))[:, :, :, :, net_idx] / 1e4

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def ensemble_evaluate(data_loader, model, net_idx, device, amp_autocast=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    net_caption = "One subnetwork for all. "
    if net_idx >=0:
        net_caption = "Subnet {}. ".format(net_idx)
    header = net_caption + header

    # switch to evaluation mode
    model.eval()
    # prefetcher = data_prefetcher(data_loader, net_idx)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if net_idx >= 0:
            images = preprocess_imagenet(images, block_size=(4, 4))[:, :, :, :, net_idx]

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters["logits"].update(output)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
