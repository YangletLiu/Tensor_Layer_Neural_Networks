import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torchvision import transforms, datasets

from utils import preprocess, fusing
from model import build
from record import save_record_and_draw
from data_loader import get_dataset
import math

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--model-name", default="FC4Net", type=str,
                    help="model name")
parser.add_argument("--filename", type=str, help="the record filename")
parser.add_argument("--subnums", type=int, help="the nums of datasets")
parser.add_argument("--l_idx", default=0, type=int,
                    help="the index of the first network")
parser.add_argument("--r_idx", default=0, type=int,
                    help="the index of the last network")
parser.add_argument("--device", default="cuda:0", type=str, help="device (Use cuda or cpu Default: cuda)")
parser.add_argument("-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
parser.add_argument("--wd","--weight-decay", default=0.0, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay",)
parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
parser.add_argument("--scheduler", default=None, type=str, help="the lr scheduler")
parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("-j", "--workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 16)")
parser.add_argument("--decom", action="store_true", help="low rank decompose the net")
parser.add_argument("--trans", default=None, help="the transform domain")
parser.add_argument("--split", default=None, type=str, help="method of split datasets")
parser.add_argument("--geo-p", default=0., type=float, help="the p of geo fusing method")
parser.add_argument("--dataset", default="MNIST", type=str)
args = parser.parse_args()



num_nets = args.r_idx - args.l_idx
if num_nets:
    blocks = int(math.sqrt(num_nets))

loss_format_str = "%.4f, " * num_nets
acc_format_str = "%.2f%%, " * num_nets
test_loss_content_str = ""
acc_content_str = ""
best_acc_content_str = ""
for __ in range(num_nets):
    test_loss_content_str += "test_loss[{}], ".format(__)
    acc_content_str += "acc[{}], ".format(__)
    best_acc_content_str += "best_acc[{}], ".format(__)

device = args.device if torch.cuda.is_available() else "cpu"
batch_size = args.batch_size

trainset, testset = get_dataset(args.dataset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

num_train = len(trainset)
num_test = len(testset)
########################### train and test functions ##################
def test(epoch, net, criterion, best_acc, test_acc_list, test_loss_list):
    net.eval()
    test_loss = 0
    correct = 0     # the number of correctly classified images
    total = 0       # the number of images

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        acc = 100.* correct / total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc: %.2f%%   " %(epoch + 1, loss.item(), acc))

        if acc > best_acc:
            best_acc = acc
            # Save checkpoint when best model
            checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_acc": best_acc,
                    }
            torch.save(checkpoint, args.model_name+"_mnist.pth.tar")
        test_acc_list.append(acc)
        test_loss_list.append(test_loss / total)
    return best_acc


def train(num_epochs, net):
    net = net.to(device)
    # initialize some metrices
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = 0.

    start_time = time.time()

    lr0 = args.lr
    current_lr = lr0
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr0, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=args.weight_decay)

    if args.scheduler == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.scheduler == "cos":
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min)



    try:
        for epoch in range(num_epochs):
            net.train()
            train_loss = 0      # training loss at this epoch
            correct = 0         # the number of correctly classified images
            total = 0           # the number of images

            print("\n=> Training Epoch #%d, LR=%.4f" %(epoch+1, current_lr))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                optimizer.zero_grad()
                outputs = net(inputs)               # Forward Propagation
                loss = criterion(outputs, targets)  # Loss
                loss.backward()                     # Backward Propagation
                optimizer.step()                    # Optimizer update

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

                sys.stdout.write('\r')
                sys.stdout.write("| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%   "
                        %(epoch+1, num_epochs, batch_idx+1,
                          (len(trainset) // batch_size) + 1, loss.item(), 100. * correct / total))
                sys.stdout.flush()

            if args.scheduler is not None:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0]

            best_acc = test(epoch, net, criterion, best_acc, test_acc_list, test_loss_list)
            train_acc_list.append(100. * correct / total)
            train_loss_list.append(train_loss / total)
            now_time = time.time()
            print("| Best Acc: %.2f%% "%(best_acc))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    print("\nBest training accuracy overall: %.3f%%"%(best_acc))

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


def test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list, fusing_test_acc_list, test_loss_list,
                     fusing_test_loss_list,fusing_plan, fusing_num, criterion, fusing_weight=None):
    for net in nets:
        net.eval()
    test_loss = [0] * num_nets
    correct = [0] * num_nets
    total = [0] * num_nets
    outputs = [0] * num_nets

    fusing_test_loss = [0] * fusing_num
    fusing_correct = [0] * fusing_num
    fusing_total = [0] * fusing_num
    fusing_outputs = [0] * fusing_num
    if fusing_weight == None:
        fusing_weight = [1. / num_nets] * num_nets

    with torch.no_grad():
        for batch_idx, (img, targets) in enumerate(testloader):
            img, targets = img.to(device), targets.to(device)
            img = preprocess(img, block_size=(blocks, blocks), method=args.split, num_nets=num_nets, trans=args.trans, device=device)

            for i in range(num_nets):
                outputs[i] = nets[i](img[:, :, :, :, i])
                loss = criterion(outputs[i], targets)

                test_loss[i] += loss.item()
                _, predicted = torch.max(outputs[i].data, 1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets.data).cpu().sum().item()

            #########################################
            ########### Fusing 2 networks ###########
            for plan_id in range(fusing_num):
                fusing_outputs[plan_id] = 0.
                fusing_weight_sum = 0.
                for net_idx in fusing_plan[plan_id]:
                    fusing_outputs[plan_id] += fusing_weight[net_idx] * outputs[net_idx]
                    fusing_weight_sum += fusing_weight[net_idx]
                fusing_outputs[plan_id] /= fusing_weight_sum

                fusing_loss = criterion(fusing_outputs[plan_id], targets)

                fusing_test_loss[plan_id] += fusing_loss.item()
                _, predicted = torch.max(fusing_outputs[plan_id].data, 1)
                fusing_total[plan_id] += targets.size(0)
                fusing_correct[plan_id] += predicted.eq(targets.data).cpu().sum().item()

            #########################################
            #########################################

        # Save checkpoint when best model
        acc = [0] * num_nets
        for i in range(num_nets):
            test_loss[i] /= num_test
            acc[i] = 100. * correct[i] / total[i]

        fusing_acc = [0] * fusing_num
        for i in range(fusing_num):
            fusing_test_loss[i] /= num_test
            fusing_acc[i] = 100. * fusing_correct[i] / fusing_total[i]

        print("| Validation Epoch #%d\t\t"% (epoch+1)
              + ("  Loss: [" + loss_format_str + "]" )%(eval(test_loss_content_str))
              + ("  Acc: [" + acc_format_str + "]")%(eval(acc_content_str))
            )

        print("| s [%.4f] Fusing Acc: [%.2f%%]   " % (fusing_test_loss[0], fusing_acc[0]))

        for i in range(num_nets):
            if acc[i] > best_acc[i]:
                best_acc[i] = acc[i]
                # torch.save(nets[i], "./"+args.filename+".pth".format(i))

        for i in range(fusing_num):
            if fusing_acc[i] > best_fusing_acc[i]:
                best_fusing_acc[i] = fusing_acc[i]

        test_acc_list.append(acc)
        test_loss_list.append([test_loss[i] for i in range(num_nets)])

        fusing_test_acc_list.append(fusing_acc)
        fusing_test_loss_list.append([fusing_test_loss[i] for i in range(fusing_num)])
    return best_acc, best_fusing_acc

def train_multi_nets(num_epochs, nets):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    lr0 = [args.lr] * num_nets
    fusing_plan = [list(range(num_nets))]
    fusing_num = len(fusing_plan)

    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = [0.] * num_nets

    fusing_test_acc_list, fusing_test_loss_list = [], []
    best_fusing_acc = [0.] * fusing_num

    start_time = time.time()

    optimizers = []
    for i in range(num_nets):
        nets[i] = nets[i].to(device)

        if args.opt == 'sgd':
            optimizers.append(torch.optim.SGD(nets[i].parameters(), lr=lr0[i], momentum=0.9, weight_decay=args.weight_decay))
        elif args.opt == 'adam':
            optimizers.append(torch.optim.Adam(nets[i].parameters(), lr=lr0[i], weight_decay=args.weight_decay))
        elif args.opt == "adamw":
            optimizers.append(torch.optim.AdamW(nets[i].parameters(), lr=lr0[i], weight_decay=args.weight_decay))

    if args.scheduler is not None:
        scheduler = []
        for i in range(num_nets):
            if args.scheduler == "steplr":
                scheduler.append(torch.optim.lr_scheduler.StepLR(
                    optimizers[i], step_size=args.lr_step_size, gamma=args.lr_gamma
                ))
            if args.scheduler == "cos":
                scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[i], T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
                ))

        current_lr = scheduler[0].get_last_lr()[0]

    try:
        for epoch in range(num_epochs):
            for net in nets:
                net.train()
            train_loss = [0] * num_nets
            correct = [0] * num_nets
            total = [0] * num_nets
            loss = [0] * num_nets

            print('\n=> Training Epoch #%d, LR=[%.4f, %.4f, %.4f, %.4f, ...]' % (epoch+1, current_lr, current_lr, current_lr, current_lr))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)  # GPU settings
                inputs = preprocess(inputs, block_size=(blocks, blocks), method=args.split, num_nets=num_nets, trans=args.trans, device=device)

                for i in range(num_nets):
                    optimizers[i].zero_grad()
                    outputs = nets[i](inputs[:, :, :, :, i])  # Forward Propagation
                    loss[i] = criterion(outputs, targets)  # Loss
                    loss[i].backward()  # Backward Propagation
                    optimizers[i].step()  # Optimizer update

                    train_loss[i] += loss[i].item()
                    _, predicted = torch.max(outputs.data, 1)
                    total[i] += targets.size(0)
                    correct[i] += predicted.eq(targets.data).cpu().sum().item()

                temp_loss_ary = np.array([loss[idx].item() for idx in range(num_nets)])
                temp_acc_ary = np.array([100.*correct[idx]/total[idx] for idx in range(num_nets)])
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: [%.4f ~ %.4f] mean: %.4f Acc: [%.3f%% ~ %.3f%%] mean: %.3f%%   '
                                 %(epoch+1, num_epochs, batch_idx+1, math.ceil(len(trainset)/batch_size),
                                   temp_loss_ary.min(), temp_loss_ary.max(), temp_loss_ary.mean(),
                                   temp_acc_ary.min(), temp_acc_ary.max(), temp_acc_ary.mean())
                                )
                sys.stdout.flush()
            if scheduler:
                for i in range(num_nets):
                    scheduler[i].step()
                current_lr = optimizers[0].param_groups[0]["lr"]

            fusing_weight = None
            if args.geo_p:
                fusing_weight = fusing(num_nets, args.geo_p, train_loss)
                fusing_weight[2] = 0.0

            best_acc, best_fusing_acc = test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list,
                                        fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_plan, fusing_num, criterion, fusing_weight=fusing_weight)

            train_acc_list.append([100.*correct[i]/total[i] for i in range(num_nets)])
            train_loss_list.append([train_loss[i] / num_train for i in range(num_nets)])
            now_time = time.time()

            print(("| Best Acc: [" + acc_format_str + "]")%(eval(best_acc_content_str)))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))

    except KeyboardInterrupt:
        pass

    print(("\nBest training accuracy overall: [" + acc_format_str + "]")%(eval(best_acc_content_str)))

    os.makedirs("./fc_models", exist_ok=True)
    for i in range(num_nets):
        torch.save(nets[i], "./"+args.filename+"_{}.pth".format(i))

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, fusing_test_loss_list, fusing_test_acc_list, fusing_plan, fusing_num


def main():
    if num_nets == 0:
        net = build(args.model_name, decomp=args.decom)
        print(net)
        train_loss, train_acc, test_loss, test_acc = train(args.epochs, net)
        save_record_and_draw(train_loss, train_acc, test_loss, test_acc, model_name=args.model_name)
    else:
        nets = []
        for _ in range(num_nets):
            nets.append(build(args.model_name, num_nets, decomp=False))
        train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_, fusing_plan, fusing_num = train_multi_nets(args.epochs, nets)
        save_record_and_draw(train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_, num_nets, fusing_plan, fusing_num, args.filename)

if __name__ == "__main__":
    main()