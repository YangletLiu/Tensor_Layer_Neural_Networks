######################### 0. import packages #############################
import time
import torch
import torchvision.models
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import preprocess, fusing
import os
import argparse
import data_loader
########################## 1. load data ####################################

transform_test = transforms.Compose([
    transforms.ToTensor()
])

total_num_nets = 36
parser = argparse.ArgumentParser(description="ImageNet Training")
parser.add_argument("--l_idx", default=0, type=int,
                    help="the index of the first network")
parser.add_argument("--r_idx", default=4, type=int,
                    help="the index of the last network")
parser.add_argument("--net_idx", default="0", type=str, help="subnetwork idx")
parser.add_argument("--premodel", default="best", type=str)
parser.add_argument("--checkpoint_path", default="spectral_resnet50_subx_best.pth", type=str)
parser.add_argument("--ensemble", default="geo", type=str)
parser.add_argument("--geo", default=0.3, type=float)
parser.add_argument("--models_path", default=".", type=str)
parser.add_argument("--device", default="7", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch", default=512, type=int)
parser.add_argument("--pretrained", default=False, type=bool)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--filename", default=os.path.basename(__file__)[:-3], type=str)
parser.add_argument("--blocks", default=4, type=int, help="number of sub-dataset")
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--record", default=True, action='store_true')
parser.add_argument("--draw", default=False, action='store_true')


args = parser.parse_args()

device = 'cuda:'+args.device if torch.cuda.is_available() else "cpu"

head_idx = args.l_idx
tail_idx = args.r_idx
num_nets = tail_idx - head_idx

# data_root = "/xfs/colab_space/yanglet/imagenet21k/" if args.dataset == "ImageNet-21K" else "/xfs/imagenet/"

preprocessing = data_loader.ClassificationPresetEval(
                crop_size=224, resize_size=232
            )

testset = torchvision.datasets.CIFAR10(
            "/xfs/home/tensor_zy/zhangjie/datasets", train=False,
            transform=preprocessing,
        )

num_test = len(testset)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch, num_workers=args.workers, pin_memory=True
)

########################### 2. define model ##################################
# build model
def build(decomp=False):
    print('==> Building model..')
    # net = vgg16()
    net = torchvision.models.resnet50(num_classes=10)
    if decomp:
        raise ("No Tensor Neural Network decompostion implementation.")
    print('==> Done')
    return net


########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss().to(device)
fusing_plan = [list(range(num_nets))]
fusing_num = len(fusing_plan)

loss_format_str = "%.4f, " * num_nets
acc_format_str = "%.2f%%, " * num_nets
test_loss_content_str = ""
top1_acc_content_str = ""
top5_acc_content_str = ""
best_acc_content_str = ""
for __ in range(num_nets):
    test_loss_content_str += "test_loss[{}], ".format(__)
    top1_acc_content_str += "acc[{}][{}], ".format(__, 0)
    top5_acc_content_str += "acc[{}][{}], ".format(__, 1)
    best_acc_content_str += "best_acc[{}], ".format(__)


def test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list, fusing_test_acc_list, test_loss_list,
                     fusing_test_loss_list, fusing_weight=None, is_best=None):
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
            img = preprocess(img, block_size=(2, 2), device=device)

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

        print("\n| Validation Epoch #%d\t\tLoss: [%.4f, %.4f, %.4f, %.4f] Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%]   "
              %(epoch+1, test_loss[0], test_loss[1], test_loss[2], test_loss[3], acc[0], acc[1], acc[2], acc[3]))

        print("| Fusing Loss: [%.4f] Fusing Acc: [%.2f%%]   "%(fusing_test_loss[0], fusing_acc[0]))


def ensemble(nets, checkpoint_path):
    start_time = time.time()
    acc = []
    for i in range(num_nets):
        checkpoint = torch.load(checkpoint_path.replace("subx", f"sub{i}"))
        acc.append(checkpoint["acc"])
        nets[i].load_state_dict(checkpoint["model"])
        nets[i].to(device)

    print(acc)
    test_acc_list, test_loss_list = [], []

    fusing_test_acc_list, fusing_test_loss_list = [], []
    best_fusing_acc = [0.] * fusing_num

    fusing_weight = [0] * num_nets

    if args.ensemble == "avg":
        for i in range(num_nets):
            fusing_weight[i] = 1 / num_nets

    if args.ensemble == "geo":
        rank_list = np.argsort(acc)[::-1]
        p = args.geo
        for i in range(num_nets):
            fusing_weight[rank_list[i]] = p * np.power((1 - p), i)

    test_fusing_nets(100, nets, acc, best_fusing_acc, test_acc_list, fusing_test_acc_list,
                     test_loss_list, fusing_test_loss_list, fusing_weight=fusing_weight)

    now_time = time.time()

    print("Used:{}s".format(now_time - start_time))

if __name__ == "__main__":
    raw_nets = []
    for _ in range(head_idx, tail_idx):
        raw_nets.append(build(decomp=False))
    ensemble(raw_nets, args.checkpoint_path)