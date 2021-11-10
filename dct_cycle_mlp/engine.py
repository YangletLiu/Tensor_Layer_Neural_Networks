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


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    if torch.__version__ > "1.7.1":
        Vc = torch.view_as_real(torch.fft.fft(v))
    else:
        Vc = torch.rfft(v, 1, onesided=False)

    k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)).cuda()
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


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


def preprocess_imagenet(img, block_size):
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # img += (torch.rand_like(img, device=device) * (ma - mi) - mi)
    img = downsample_img(img, block_size=block_size)
    # print(img.shape)
    # exit(0)
    img = dct(img)
    return img


# class data_prefetcher():
#     def __init__(self, loader, net_idx):
#         self.loader = iter(loader)
#         self.length = len(loader)
#         self.stream = torch.cuda.Stream()
#         self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
#         self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
#         self.net_idx = net_idx
#         # With Amp, it isn't necessary to manually convert data to half.
#         # if args.fp16:
#         #     self.mean = self.mean.half()
#         #     self.std = self.std.half()
#         self.preload()

#     def preload(self):
#         try:
#             self.next_input, self.next_target = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             self.next_target = None
#             return
#         # if record_stream() doesn't work, another option is to make sure device inputs are created
#         # on the main stream.
#         # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
#         # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
#         # Need to make sure the memory allocated for next_* is not still in use by the main stream
#         # at the time we start copying to next_*:
#         # self.stream.wait_stream(torch.cuda.current_stream())
#         with torch.cuda.stream(self.stream):
#             self.next_input = self.next_input.cuda(non_blocking=True)
#             self.next_target = self.next_target.cuda(non_blocking=True)
#             # more code for the alternative if record_stream() doesn't work:
#             # copy_ will record the use of the pinned source tensor in this side stream.
#             # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
#             # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
#             # self.next_input = self.next_input_gpu
#             # self.next_target = self.next_target_gpu

#             # With Amp, it isn't necessary to manually convert data to half.
#             # if args.fp16:
#             #     self.next_input = self.next_input.half()
#             # else:
#             self.next_input = self.next_input.float()
#             self.next_input = self.next_input.sub_(self.mean).div_(self.std)
#             self.next_input = preprocess_imagenet(self.next_input, block_size=(4, 4))
#             if self.net_idx >=0:
#                 self.next_input = self.next_input[:, :, :, :, self.net_idx]

#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         input = self.next_input
#         target = self.next_target
#         if input is not None:
#             input.record_stream(torch.cuda.current_stream())
#         if target is not None:
#             target.record_stream(torch.cuda.current_stream())
#         self.preload()
#         return input, target

#     def __len__(self):
#         return self.length


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
