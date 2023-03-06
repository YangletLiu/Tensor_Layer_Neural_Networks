######################### 0. import packages #############################
import time
import torch
import torchvision.models
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
import os
import argparse
from model import resnet34
########################## 1. load data ####################################

transform_test = transforms.Compose([
    transforms.ToTensor()
])

total_num_nets = 36
parser = argparse.ArgumentParser(description="ImageNet Training")
parser.add_argument("--l_idx", default=0, type=int,
                    help="the index of the first network")
parser.add_argument("--r_idx", default=2, type=int,
                    help="the index of the last network")
parser.add_argument("--net_idx", default="0", type=str, help="subnetwork idx")
parser.add_argument("--premodel", default="best", type=str)
parser.add_argument("--checkpoint_path", default="resnet34_newsubx_best.pth", type=str)
parser.add_argument("--ensemble", default="geo", type=str)
parser.add_argument("--geo", default=0.3, type=float)
parser.add_argument("--models_path", default="./dct_resnet_models", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch", default=1024, type=int)
parser.add_argument("--pretrained", default=False, type=bool)


parser.add_argument("--dataset", default="ImageNet-21K", type=str)
parser.add_argument("--filename", default=os.path.basename(__file__)[:-3], type=str)
parser.add_argument("--blocks", default=16, type=int, help="number of sub-dataset")
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--record", default=True, action='store_true')
parser.add_argument("--draw", default=False, action='store_true')


args = parser.parse_args()

device = 'cuda:'+args.device if torch.cuda.is_available() else "cpu"

head_idx = args.l_idx
tail_idx = args.r_idx
num_nets = tail_idx - head_idx

data_root = "/xfs/colab_space/yanglet/imagenet21k/" if args.dataset == "ImageNet-21K" else "/xfs/imagenet/"

testset = []
testloader = []
for i in range(num_nets):
    testset.append(datasets.ImageFolder(root=f'/xfs/colab_space/yanglet/imagenet21k-sub{i}-notdct/' + "val", transform=transform_test))
    testloader.append(torch.utils.data.DataLoader(testset[i], batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True))

num_test = len(testset[0])

########################### 2. define model ##################################
# define nn module
save_models_path = args.models_path
load_models_path = args.models_path
os.makedirs(save_models_path, exist_ok=True)


def save_checkpoint(state, filename):
    dir_path = save_models_path
    torch.save(state, dir_path + '/' + filename)


def load_checkpoint(filename, net, optimizer, device):
    dir_path = load_models_path
    loc = device
    checkpoint = torch.load(dir_path + filename, map_location=loc)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Successfully loaded a model with accuracy: {}".format(checkpoint["best_acc1"]))


def dct(x, device, norm=None):
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

    k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)).to(device)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def preprocess_imagenet(img, block_size=(6, 6), device=device):
    img = downsample_img(img, block_size=block_size, total_num_nets=36)
    print("shape:", img.shape)
    for i in range(img.shape[0]):
        print("shape:", img[i].shape)
        img[i] = dct(img[i], device)
    img = img[:, :, :, :, head_idx:tail_idx]
    return img

def downsample_img(img, block_size, total_num_nets):
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
            components.append(img[:, :, row::row_step, col::col_step])
    img = torch.stack(components, dim=-1)

    # for i in range(row_step * col_step):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))
    return img
######################## 3. build model functions #################

# build model
def build(decomp=False):
    print('==> Building model..')
    # net = vgg16()
    net = resnet34(num_classes = 10450)
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
    correct = [[0, 0]] * num_nets
    total = [0] * num_nets
    outputs = [0] * num_nets

    fusing_test_loss = [0] * fusing_num
    fusing_correct = [[0, 0]] * fusing_num
    fusing_total = [0] * fusing_num
    fusing_outputs = [0] * fusing_num
    if fusing_weight == None:
        fusing_weight = [1. / num_nets] * num_nets

    with torch.no_grad():
        # for batch_idx, ((img_0, targets), (img_1, targets), (img_2, targets), (img_3, targets)) in enumerate(zip(testloader[0], testloader[1], testloader[2], testloader[3])):
        for batch_idx, ((img_0, targets), (img_1, targets)) in enumerate(zip(testloader[0], testloader[1])):

            img = [img_0.to(device), img_1.to(device)]
            # img = [img_0.to(device), img_1.to(device), img_2.to(device), img_3.to(device)]
            targets = targets.to(device)
            # img, targets = img.to(device), targets[0].to(device)
            for i in range(num_nets):
                outputs[i] = nets[i](img[i])
                loss = criterion(outputs[i], targets)

                test_loss[i] += loss.item()
                _, pred = outputs[i].data.topk(5, 1, True, True)
                pred = pred.t()
                cor = pred.eq(targets[None])

                total[i] += targets.size(0)
                correct[i][0] += cor[:1].flatten().cpu().sum().item()
                correct[i][1] += cor[:5].flatten().cpu().sum().item()


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
                _, pred = fusing_outputs[plan_id].data.topk(5, 1, True, True)
                fusing_total[plan_id] += targets.size(0)
                pred = pred.t()
                cor = pred.eq(targets[None])
                fusing_correct[plan_id][0] += cor[:1].flatten().cpu().sum().item()
                fusing_correct[plan_id][1] += cor[:5].flatten().cpu().sum().item()

            #########################################
            #########################################

        # Save checkpoint when best model
        acc = [[0, 0.]] * num_nets
        for i in range(num_nets):
            test_loss[i] /= num_test
            acc[i][0] = 100. * correct[i][0] / total[i]
            acc[i][1] = 100. * correct[i][1] / total[i]

        fusing_acc = [[0, 0]] * fusing_num
        for i in range(fusing_num):
            fusing_test_loss[i] /= num_test
            fusing_acc[i][0] = 100. * fusing_correct[i][0] / fusing_total[i]
            fusing_acc[i][1] = 100. * fusing_correct[i][1] / fusing_total[i]

        print("| Validation Epoch #%d\t\t" % (epoch + 1)
              + ("  Loss: [" + loss_format_str + "]") % (eval(test_loss_content_str))
              + ("  top1-Acc: [" + acc_format_str + "]") % (eval(top1_acc_content_str))
              + ("  top5-Acc: [" + acc_format_str + "]") % (eval(top5_acc_content_str))
              )

        print("| fusing Loss: [%.4f], Fusing Acc: top1-[%.2f%%],top5-[%.2f%%]" % (fusing_test_loss[0], fusing_acc[0][0], fusing_acc[0][1]))

        # for i in range(num_nets):
        #     if acc[i] > best_acc[i]:
        #         best_acc[i] = acc[i]
        #         if is_best is not None:
        #             is_best[i] = True

        # for i in range(fusing_num):
        #     if fusing_acc[i] > best_fusing_acc[i]:
        #         best_fusing_acc[i] = fusing_acc[i]
        #
        # test_acc_list.append(acc)
        # test_loss_list.append([test_loss[i] for i in range(num_nets)])
        #
        # fusing_test_acc_list.append(fusing_acc)
        # fusing_test_loss_list.append([fusing_test_loss[i] for i in range(fusing_num)])
    # return best_acc, best_fusing_acc

def ensemble(nets, checkpoint_path):
    acc = []
    start_time = time.time()
    for i in range(num_nets):
        checkpoint = torch.load(checkpoint_path.replace("subx", f"sub{i}"))
        acc.append(checkpoint['acc'])
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
    print(raw_nets[0])
    ensemble(raw_nets, args.checkpoint_path)
