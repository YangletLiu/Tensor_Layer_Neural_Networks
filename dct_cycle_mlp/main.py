# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from contextlib import suppress

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, preprocess_imagenet, accuracy
from losses import DistillationLoss
from samplers import RASampler
import cycle_mlp
import utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from timm.utils import ApexScaler
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from fvcore.nn import flop_count, parameter_count, FlopCountAnalysis, flop_count_table
    from utils import sfc_flop_jit
    has_fvcore = True
except ImportError:
    has_fvcore = False


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--net_idx', default=-1, type=int, help='subnetwork number')
    parser.add_argument("--ens_val", action="store_true")
    parser.add_argument("--ens_net", type=str)
    parser.add_argument("--load_dir", type=str)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # EMA (Exponential Moving Average)
    # When training a model, it is often beneficial to maintain moving averages of the trained parameters.
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
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

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
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

    # Distillation parameters
    parser.add_argument('--teacher-model', default='none', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "none"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--mcloader', action='store_true', default=False, help='whether use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./cycle_mlp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # custom parameters
    parser.add_argument('--flops', action='store_true', help='whether calculate FLOPs of the model')
    parser.add_argument('--no_amp', action='store_true', help='not using amp')
    return parser


def fast_collate(batch, memory_format):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if not args.no_amp:  # args.amp: Default  use AMP
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
            args.apex_amp = False
        elif has_apex:
            args.native_amp = False
            args.apex_amp = True
        else:
            raise ValueError("Warning: Neither APEX or native Torch AMP is available, using float32."
                             "Install NVIDA apex or upgrade to PyTorch 1.6")
    else:
        args.apex_amp = False
        args.native_amp = False
        print("Do not use amp.")
    if args.apex_amp and has_apex:
        use_amp = 'apex'
        print("Use apex amp.")
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
        print("Use native amp.")
    elif args.apex_amp or args.native_amp:
        print("Warning: Neither APEX or native Torch AMP is available, using float32. "
                       "Install NVIDA apex or upgrade to PyTorch 1.6")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # memory_format = torch.contiguous_format
    # collate_fn = lambda b: fast_collate(b, memory_format)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # collate_fn=collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        # collate_fn=collate_fn
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=56
    )

    if args.flops:
        if not has_fvcore:
            print("Please install fvcore first for FLOPs calculation.")
        else:
            # Set model to evaluation mode for analysis.
            model_mode = model.training
            model.eval()
            fake_input = torch.rand(1, 3, 224, 224)
            flops_dict, *_ = flop_count(model, fake_input,
                                        supported_ops={"torchvision::deform_conv2d": sfc_flop_jit})
            count = sum(flops_dict.values())
            model.train(model_mode)
            print('=' * 30)
            print("fvcore MAdds: {:.3f} G".format(count))

    if args.finetune:  # do not finetune by default
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        print('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print('Using native Torch AMP. Training in mixed precision.')
    else:
        print('AMP not enabled. Training in float32.')

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print('=' * 30)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:  # args.mixup = 0.8 by default: use mixup
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':  # do not use distillation by default
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    net_idx = args.net_idx

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, net_idx, device, amp_autocast=amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    num_epochs = args.epochs - args.start_epoch
    print(f"Start training for {num_epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    save_freq = 10
    is_best = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        is_best = False

        train_stats = train_one_epoch(
            model, net_idx, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            amp_autocast=amp_autocast,
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, net_idx, device, amp_autocast=amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if test_stats["acc1"] > max_accuracy:
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            is_best = True
        print(f'Max accuracy: {max_accuracy:.2f}%\n')

        if args.output_dir:
            if ((epoch + 1) % save_freq == 0) or (epoch == 0):
                if net_idx >= 0:
                    checkpoint_paths = [output_dir / 'regular_model_{}.pth'.format(net_idx)]
                else:
                    checkpoint_paths = [output_dir / 'regular_model.pth']
                print("Regularly saving models...")
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, checkpoint_path)
            if is_best:
                if net_idx >= 0:
                    checkpoint_paths = [output_dir / 'best_model_{}.pth'.format(net_idx)]
                else:
                    checkpoint_paths = [output_dir / 'best_model.pth']
                print("Saving best model...")
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, checkpoint_path)

        if args.output_dir and utils.is_main_process():
            now_time = time.time()
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        "used_time": now_time - start_time,
                        "est": (now_time-start_time)/(epoch-args.start_epoch+1)*(num_epochs-(epoch-args.start_epoch)-1)}
            print("Used:{}s \t EST: {}s\n".format(now_time-start_time, (now_time-start_time)/(epoch-args.start_epoch+1)*(num_epochs-(epoch-args.start_epoch)-1)))
            if net_idx >= 0:
                log_filename = "log_{}.txt".format(net_idx)
            else:
                log_filename = "log.txt"
            with (output_dir / log_filename).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if net_idx >= 0:
        print("Subnet_{} training time {}".format(net_idx, total_time_str))
    else:
        print('Training time {}'.format(total_time_str))


def ensemble(args):
    # utils.init_distributed_mode(args)
    net_indices = list(np.sort(np.unique([int(i) for i in args.ens_net.split(",")])))
    num_nets = len(net_indices)
    # quota = []
    # for i in range(utils.get_rank(), len(ensemble_nets), utils.get_world_size()):
    #     quota.append(i)

    print(args)

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if not args.no_amp:  # args.amp: Default  use AMP
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
            args.apex_amp = False
        elif has_apex:
            args.native_amp = False
            args.apex_amp = True
        else:
            raise ValueError("Warning: Neither APEX or native Torch AMP is available, using float32."
                             "Install NVIDA apex or upgrade to PyTorch 1.6")
    else:
        args.apex_amp = False
        args.native_amp = False
        print("Do not use amp.")
    if args.apex_amp and has_apex:
        use_amp = 'apex'
        print("Use apex amp.")
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
        print("Use native amp.")
    elif args.apex_amp or args.native_amp:
        print("Warning: Neither APEX or native Torch AMP is available, using float32. "
                       "Install NVIDA apex or upgrade to PyTorch 1.6")

    # fix the seed for reproducibility

    cudnn.benchmark = True

    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        # collate_fn=collate_fn
    )

    print(f"Creating model: {args.model}")
    nets = []
    for _ in range(num_nets):
        nets.append(
            create_model(
                args.model,
                pretrained=False,
                num_classes=1000,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=56
                )
            )
        # nets[-1].to(device)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing

    # model_without_ddp = model
    # if args.distributed:
    #     if has_apex and use_amp != 'native':
    #         # Apex DDP preferred unless native amp is activated
    #         model = ApexDDP(model, delay_allreduce=True)
    #     else:
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    # print('=' * 30)

    # lr_scheduler, _ = create_scheduler(args, optimizer)

    load_dir = Path(args.load_dir)
    for _ in range(num_nets):
        net_idx = net_indices[_]
        checkpoint = torch.load(load_dir/"best_model_{}.pth".format(net_idx), map_location='cpu')
        nets[_].load_state_dict(checkpoint["model"])
        print("Successfully load subnet-{}".format(net_idx))
        nets[_] = nets[_].to(device)
        nets[_].eval()

    # weights = np.array([1.] * num_nets)

    # weights = np.array([0.006870775814056397, 0.00772176059782505, 0.009846172868013382])

    p = 0.5
    weights = np.array([p * np.power((1 - p), i) for i in range(num_nets)])

    weights = weights / weights.sum()

    output_list = [0] * num_nets
    loss_list = [0] * num_nets
    fusing_loss = 0.
    total = [0] * num_nets
    correct = [0] * num_nets
    fusing_correct = 0
    fusing_total = 0

    start_time = time.time()
    header = 'Test:'

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for (images, target) in data_loader_val:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            images = preprocess_imagenet(images, block_size=(4, 4))

            for i in range(num_nets):
                net_idx = net_indices[i]
                output = nets[i](images[:, :, :, :, net_idx])
                loss = criterion(output, target)

                output_list[i] = output.clone()
                loss_list[i] += loss.item()
                _, predicted = torch.max(output.data, 1)
                total[i] += target.size(0)
                correct[i] += predicted.eq(target.data).cpu().sum().item()

            fusing_output = 0
            for _  in range(num_nets):
                fusing_output += weights[_] * output_list[_]

            fusing_loss += criterion(fusing_output, target)
            _, predicted = torch.max(fusing_output.data, 1)
            fusing_total += target.size(0)
            fusing_correct += predicted.eq(target.data).cpu().sum().item()

    acc_list = [0.] * num_nets
    for i in range(num_nets):
        loss_list[i] /= total[i]
        acc_list[i] = 100. * correct[i] / total[i]
    fusing_loss /= fusing_total
    fusing_acc1 = 100. * fusing_correct / fusing_total

    end_time = time.time()
    print("net indices: {}".format(net_indices))
    print("separat acc1: {}".format(acc_list))
    print("separarte loss: {}".format(loss_list))
    print("fusing acc1: {}".format(fusing_acc1))
    print("fusing loss: {}".format(fusing_loss))
    print("Time cost: {}s".format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    local_args = parser.parse_args()
    if local_args.output_dir:
        Path(local_args.output_dir).mkdir(parents=True, exist_ok=True)
    if local_args.ens_val:  # do ensemble validation
        ensemble(local_args)
    else:
        main(local_args)


############### running command ########################

'''
On dgx64:

CUDA_VISIBLE_DEVICES=5 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25199 --use_env main.py --model CycleMLP_B5 --batch-size 256 --num_workers 16 --data-path /xfs/imagenet/ --net_idx 0

python -u -m torch.distributed.launch --nproc_per_node=5 --master_port 25197 --use_env main.py --model CycleMLP_B5 --batch-size 1024 --num_workers 24 --data-path /xfs/imagenet/ --net_idx 0

python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25197 --use_env main.py --model CycleMLP_B5 --batch-size 1024 --num_workers 24 --data-path /xfs/imagenet/ --net_idx 0 --resume ./cycle_mlp/best_model_0.pth


fusing test:
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25196 --use_env main.py --model CycleMLP_B5 --batch-size 128 --num_workers 16 --data-path /xfs/imagenet/ --ens_val --load_dir ./cycle_mlp --ens_net 0,1,2


On cluster:

python main.py --model CycleMLP_B5 --batch_size 128 --num_workers 10 --data_path /colab_space/imagenet

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 25198 --use_env main.py --model CycleMLP_B5 --batch-size 128 --num_workers 16 --data-path /colab/imagenet/

'''