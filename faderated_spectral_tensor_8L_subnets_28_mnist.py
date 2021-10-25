######################### 0. import packages #############################
import time
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
import os
import argparse


########################## 1. load data ####################################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.1307,), (0.3081,))
                              ])

transform_test = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.1307,), (0.3081,))
                              ])

total_num_nets = 28
parser = argparse.ArgumentParser(description="ImageNet Training")
parser.add_argument("--l_idx", default=0, type=int, 
                    help="the index of the first network")
parser.add_argument("--r_idx", default=total_num_nets, type=int, 
                    help="the index of the last network")
args = parser.parse_args()

head_idx = args.l_idx
tail_idx = args.r_idx
num_nets = tail_idx - head_idx


batch_size = 128
data_root = "../datasets"
trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
num_train = len(trainset)

testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
num_test = len(testset)


########################### 2. define model ##################################
# define nn module
class FC8Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3,
                 n_hidden_4, n_hidden_5, n_hidden_6, n_hidden_7, out_dim):
        super(FC8Net, self).__init__()
        # layer1
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_5),
            nn.ReLU(True)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(n_hidden_5, n_hidden_6),
            nn.ReLU(True)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(n_hidden_6, n_hidden_7),
            nn.ReLU(True)
        )
        self.layer8 = nn.Sequential(
            nn.Linear(n_hidden_7, out_dim),
        )
    # forward
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
# 0 ~ 27 subnets
# save_models_path = "./faderal_spec_high_low_models/"
# load_models_path = "./faderal_spec_high_low_models/"
# 0 ~ 14 subnets
save_models_path = "./faderal_spec_low_models/"
load_models_path = "./faderal_spec_low_models/"
# 14 ~ 27 subnets
# save_models_path = "./faderal_spec_high_models/"
# load_models_path = "./faderal_spec_high_models/"
os.makedirs(save_models_path, exist_ok=True)
def save_checkpoint(state, filename, dir_path=save_models_path):
    dir_path = save_models_path
    torch.save(state, dir_path + filename)


def load_checkpoint(filename, net, optimizer, device, dir_path=load_models_path):
    loc = device
    checkpoint = torch.load(dir_path + filename, map_location=loc)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Successfully loaded a model with accuracy: {}".format(checkpoint["best_acc1"]))

######################## 3. build model functions #################
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

    k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)).to(device)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (torch.arange(x_shape[-1], dtype=X.dtype)[None, :] * np.pi / (2 * N)).to(device)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    if torch.__version__ > "1.7.1":
        v = torch.fft.ifft(torch.view_as_complex(V)).real
    else:
        v = torch.irfft(V, 1, onesided=False)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def show_mnist_fig(im, file_name="show_image.png"):
    im = np.array(im)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.cla()
    plt.axis('off')
    plt.imshow(im, cmap='gray')
    # plt.show()
    plt.savefig(file_name)  # 保存成文件
    plt.close()
    return


def downsample_img(img, block_size):
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


def preprocess_mnist(img, block_size):
    mi, ma = -0.4242, 2.8215
    # img += (torch.rand_like(img, device=device) * (ma - mi) - mi)
    img = downsample_img(img, block_size=block_size)
    # print(img.shape)
    # exit(0)
    img = dct(img)
    img = img[:, :, :, :, head_idx:tail_idx]
    return img


# build model
def build(decomp=False):
    print('==> Building model..')
    full_net = FC8Net(28, 28, 28, 28, 28, 28, 28, 28, 10)
    if decomp:
        raise("No Tensor Neural Network decompostion implementation.")
    print('==> Done')
    return full_net


########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss().to(device)
lr0 = [0.001] * num_nets
fusing_plan = [list(range(28))]
fusing_num = len(fusing_plan)

loss_format_str = "%.4f, " * num_nets
acc_format_str = "%.2f%%, " * num_nets
test_loss_content_str = ""
acc_content_str = ""
best_acc_content_str = ""
for __ in range(num_nets):
    test_loss_content_str += "test_loss[{}], ".format(__)
    acc_content_str += "acc[{}], ".format(__)
    best_acc_content_str += "best_acc[{}], ".format(__)
    

def query_lr(epoch):
    lr = lr0
    return lr


def set_lr(optimizer, epoch):
    current_lr = query_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list, fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_weight=None, is_best=None):
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
            img = preprocess_mnist(img, block_size=(4, 7))

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

        print("| Validation Epoch #%d\t\t"%(epoch+1)
              + ("  Loss: [" + loss_format_str + "]" )%(eval(test_loss_content_str))
              + ("  Acc: [" + acc_format_str + "]")%(eval(acc_content_str))
            )

        print("| Fusing Loss: [%.4f, %.4f]\t"%(fusing_test_loss[0], fusing_test_loss[1])
              +"Fusing Acc: [%.2f%%, %.2f%%]  "%(fusing_acc[0], fusing_acc[1]))

        for i in range(num_nets):
            if acc[i] > best_acc[i]:
                best_acc[i] = acc[i]
                # torch.save(nets[i], "./multi_cnn_8_channel_{}.pth".format(i))

        for i in range(fusing_num):
            if fusing_acc[i] > best_fusing_acc[i]:
                best_fusing_acc[i] = fusing_acc[i]
        
        test_acc_list.append(acc)
        test_loss_list.append([test_loss[i] for i in range(num_nets)])

        fusing_test_acc_list.append(fusing_acc)
        fusing_test_loss_list.append([fusing_test_loss[i] for i in range(fusing_num)])
    return best_acc, best_fusing_acc


def test_multi_nets(epoch, nets, best_acc, test_acc_list, test_loss_list, is_best=None):
    for net in nets:
        net.eval()
    test_loss = [0] * num_nets
    correct = [0] * num_nets
    total = [0] * num_nets
    outputs = [0] * num_nets

    with torch.no_grad():
        for batch_idx, (img, targets) in enumerate(testloader):
            img, targets = img.to(device), targets.to(device)
            img = preprocess_mnist(img, block_size=(4, 7))

            for i in range(num_nets):
                outputs[i] = nets[i](img[:, :, :, :, i])
                loss = criterion(outputs[i], targets)

                test_loss[i] += loss.item()
                _, predicted = torch.max(outputs[i].data, 1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets.data).cpu().sum().item()
        # Save checkpoint when best model
        acc = [0] * num_nets
        for i in range(num_nets):
            test_loss[i] /= num_test
            acc[i] = 100. * correct[i] / total[i]
        print("| Validation Epoch #%d\t\t"%(epoch+1)
              + ("  Loss: [" + loss_format_str + "]" )%(eval(test_loss_content_str))
              + ("  Acc: [" + acc_format_str + "]")%(eval(acc_content_str))
            )

        for i in range(num_nets):
            if acc[i] > best_acc[i]:
                best_acc[i] = acc[i]
                is_best[i] = 1
        test_acc_list.append(acc)
        test_loss_list.append([test_loss[i] for i in range(num_nets)])

    return best_acc


# Training
def train_multi_nets(num_epochs, nets):
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = [0.] * num_nets

    fusing_test_acc_list, fusing_test_loss_list = [], []
    best_fusing_acc = [0.] * fusing_num

    start_time = time.time()

    optimizers = []

    for i in range(num_nets):
        nets[i] = nets[i].to(device)
        # optimizers.append(torch.optim.SGD(nets[i].parameters(), lr=lr0[i], momentum=0.9))
        optimizers.append(torch.optim.Adam(nets[i].parameters(), lr=lr0[i]))

    current_lr = lr0

    try:
        for epoch in range(num_epochs):
            for net in nets:
                net.train()
            train_loss = [0] * num_nets
            correct = [0] * num_nets
            total = [0] * num_nets
            loss = [0] * num_nets
            print("\nFor subnets {}~{}: ".format(head_idx, tail_idx-1))
            print('=> Training Epoch #%d, LR=[%.4f]'%(epoch+1, current_lr[0]))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)  # GPU settings
                inputs = preprocess_mnist(inputs, block_size=(4, 7))

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
            fusing_weight = [0] * num_nets
            # for i in range(num_nets):
            #     fusing_weight[i] = 1
            p = 0.3
            rank_list = np.argsort(train_loss)
            fusing_weight = [0] * num_nets
            for i in range(num_nets):
                fusing_weight[rank_list[i]] = p * np.power((1 - p), i)
                # fusing_weight[i] = 1 / train_loss[i]

            is_best = torch.zeros(size=(num_nets,))
            best_acc = test_multi_nets(epoch, nets, best_acc, test_acc_list, test_loss_list, is_best=is_best)

            train_acc_list.append([100.*correct[i]/total[i] for i in range(num_nets)])
            train_loss_list.append([train_loss[i] / num_train for i in range(num_nets)])
            now_time = time.time()

            print(("| Best Acc: [" + acc_format_str + "]")%(eval(best_acc_content_str)))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))

            if ((epoch + 1) % 10 == 0) or (epoch == 0):
                print("Regularly saving models...")
                for net_idx in range(num_nets):
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': nets[net_idx].state_dict(),
                            'best_acc1': best_acc[net_idx],
                            "optimizer": optimizers[net_idx]
                        },
                        filename="regular_model_{}.pth.tar".format(head_idx + net_idx)
                    )

            for net_idx in range(num_nets):
                if is_best[net_idx]:
                    print("Saving best model-{}...".format(head_idx + net_idx))
                    save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'state_dict': nets[net_idx].state_dict(),
                                'best_acc1': best_acc[net_idx],
                                "optimizer": optimizers[net_idx]
                            },
                            filename="best_model_{}.pth.tar".format(head_idx + net_idx)
                        )

    except KeyboardInterrupt:
        pass

    print(("\nBest training accuracy overall: [" + acc_format_str + "]")%(eval(best_acc_content_str)))

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, fusing_test_loss_list, fusing_test_acc_list


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc, fusing_test_loss, fusing_test_acc):
    # write csv
    with open('faderal_spectral_tensor_8L_subnets_28_testloss.csv','w',newline='',encoding='utf-8') as f:
        f_csv = csv.writer(f)

        f_csv.writerow(["Test Acc:"])
        for idx in range(len(test_acc)):
            f_csv.writerow([idx + 1] + test_acc[idx])

        f_csv.writerow(["Test Loss:"])
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + test_loss[idx])
            
        f_csv.writerow(["Train Acc"])
        for idx in range(len(train_acc)):
            f_csv.writerow([idx + 1] + train_acc[idx])

        f_csv.writerow(["Train Loss"])
        for idx in range(len(train_loss)):
            f_csv.writerow([idx + 1] + train_loss[idx])

    # draw picture
    test_acc = np.array(test_acc)
    test_loss = np.array(test_loss)
    fusing_test_acc = np.array(fusing_test_acc)
    fusing_test_loss = np.array(fusing_test_loss)
    train_acc = np.array(train_acc)
    train_loss = np.array(train_loss)

    plt.cla()
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title('faderal-spectral-tensor-8L-subnets-28 Loss on MNIST ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_loss[:, i])), test_loss[:, i], label='TestLoss_{}'.format(i+1),linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_loss[:, i])), train_loss[:, i], label='TrainLoss_{}'.format(i+1),linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('faderal-spectral-tensor-8L-subnets-28 Accuracy on MNIST ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_acc[:, i])), test_acc[:, i], label='TestAcc_{}'.format(i+1),linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_acc[:, i])), train_acc[:, i], label='TrainAcc_{}'.format(i+1),linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./faderal_spectral_tensor_8L_subnets_28.jpg')


if __name__ == "__main__":
    raw_nets = []
    for _ in range(head_idx, tail_idx):
        raw_nets.append(build(decomp=False))
    print(raw_nets[0])
    train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_ = train_multi_nets(100, raw_nets)
    save_record_and_draw(train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_)
