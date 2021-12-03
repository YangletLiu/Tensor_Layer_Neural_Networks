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

num_nodes = 3
num_nets = 3
total_num_nets = 28
num_spectrals = 28
batch_size = 128
trainset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
num_train = len(trainset)

testset = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform_test)
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

def copy_parameters(dst_net, src_net):
    for param1, param2 in zip(dst_net.parameters(), src_net.parameters()):
        param1.data = param2.data.clone()
    return

save_models_path = "./faderated_fc_3_nodes_models/"
load_models_path = "./faderated_fc_3_nodes_models/"
os.makedirs(save_models_path, exist_ok=True)
def save_checkpoint(state, filename, dir_path=save_models_path):
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
    assert num_spectrals == row_step * col_step, "the number of downsampled images is not equal to the number of num_spectrals"
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
    block_row_per_grid = m_ // row_per_block
    block_col_per_grid = n_ // col_per_block
    assert num_spectrals == block_row_per_grid * block_col_per_grid, "the number of downsampled images is not equal to the number of num_spectrals"
    assert m_ % row_per_block == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_per_block == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # show_mnist_fig(img[0, 0, :, :], "split_image_seg{}.png".format(num_nets))

    components = []
    for row_block_idx in range(block_row_per_grid):
        for col_block_idx in range(block_col_per_grid):
            components.append(img[:, 
                                  :, 
                                  row_block_idx * row_per_block : row_block_idx * row_per_block + row_per_block, 
                                  col_block_idx * col_per_block : col_block_idx * col_per_block + col_per_block].unsqueeze(dim=-1))
    img = torch.cat(components, dim=-1)

    # for i in range(row_blocks * col_blocks):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))
    # print(img.shape)
    # exit(0)

    return img


def inv_block_img(img, grid_size):
    batch_, c_, row_per_block, col_per_grid, k_ = img.shape
    block_row_per_grid, block_col_per_grid = grid_size
    m_ = row_per_block * block_row_per_grid
    n_ = col_per_grid * block_col_per_grid
    assert num_spectrals == block_col_per_grid * block_row_per_grid, "the number of downsampled images is not equal to the number of num_spectrals"
    assert m_ % row_per_block == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_per_grid == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # show_mnist_fig(img[0, 0, :, :], "split_image_seg{}.png".format(num_nets))

    big_components = []
    small_components = []
    for spectral_idx in range(k_):
        small_components.append(img[:, :, :, :, spectral_idx])
        if len(small_components) == block_col_per_grid:
            big_components.append(torch.cat(small_components, dim=-1))
            small_components = []

    # print(small_components[0].shape)
    img = torch.cat(big_components, dim=-2)

    # show_mnist_fig(img[0, 0, :, :], "inverse_img.png")
    # for i in range(row_blocks * col_blocks):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))
    # print(img.shape)
    # exit(0)

    return img


def preprocess_mnist(img, block_size):
    mi, ma = -0.4242, 2.8215
    # img += (torch.rand_like(img, device=device) * (ma - mi) - mi)
    img = block_img(img, block_size=block_size)
    img = dct(img)
    return img


def inv_preprocess_mnist(img, grid_size):
    img = idct(img)
    img = inv_block_img(img, grid_size=grid_size)
    return img


# build model
def build(decomp=False):
    print('==> Building model..')
    full_net = FC8Net(784, 784, 784, 784, 784, 784, 784, 784, 10)
    if decomp:
        raise("No Tensor Neural Network decompostion implementation.")
    print('==> Done')
    return full_net


########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss().to(device)
lr0 = [0.001, 0.001, 0.001]
fusing_plan = [list(range(28))]
fusing_num = len(fusing_plan)

loss_format_str = "%.4f, " * num_nets
acc_format_str = "%.2f%%, " * num_nets
loss_content_str = ""
test_loss_content_str = ""
acc_content_str = ""
best_acc_content_str = ""
for __ in range(num_nets):
    loss_content_str += "loss[{}], ".format(__)
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


def avg_gradients(nets):
    for net_idx in range(1, num_nets):
        for dst_param, src_param in zip(nets[0].parameters(), nets[net_idx].parameters()):
            dst_param.grad.data += src_param.grad.data

    for param in nets[0].parameters():
        param.grad.data /= num_nets

    for net_idx in range(1, num_nets):
        for dst_param, src_param in zip(nets[net_idx].parameters(), nets[0].parameters()):
            dst_param.grad.data = src_param.grad.data.clone()

    return

def test_multi_nets(epoch, nets, best_acc, test_acc_list, test_loss_list, is_best=None):
    for net in nets:
        net.eval()
    test_loss = [0] * num_nets
    correct = [0] * num_nets
    total = [0] * num_nets
    outputs = [0] * num_nets

    with torch.no_grad():
        for batch_idx, (original_inputs, targets) in enumerate(testloader):
            original_inputs, targets = original_inputs.to(device), targets.to(device)
            original_inputs = preprocess_mnist(original_inputs, block_size=(4, 7))

            for net_idx in range(num_nets):
                inputs = original_inputs.clone()
                if net_idx == 0:
                    inputs = inv_preprocess_mnist(original_inputs, grid_size=(7, 4))
                elif net_idx == 1:  # only process high-frequency data
                    inputs[:, :, :, :, :num_spectrals//2] = 0
                    inputs = inv_preprocess_mnist(original_inputs, grid_size=(7, 4))
                elif net_idx == 2:  # only process low-frequency data
                    inputs[:, :, :, :, num_spectrals//2:] = 0
                    inputs = inv_preprocess_mnist(original_inputs, grid_size=(7, 4))

                outputs[net_idx] = nets[net_idx](inputs)
                loss = criterion(outputs[net_idx], targets)

                test_loss[net_idx] += loss.item()
                _, predicted = torch.max(outputs[net_idx].data, 1)
                total[net_idx] += targets.size(0)
                correct[net_idx] += predicted.eq(targets.data).cpu().sum().item()
        # Save checkpoint when best model
        acc = [0] * num_nets
        for net_idx in range(num_nets):
            test_loss[net_idx] /= num_test
            acc[net_idx] = 100. * correct[net_idx] / total[net_idx]
        print("\n| Validation Epoch #%d\t\t"%(epoch+1)
              + ("  Loss: [" + loss_format_str + "]" )%(eval(test_loss_content_str))
              + ("  Acc: [" + acc_format_str + "]")%(eval(acc_content_str))
            )

        for net_idx in range(num_nets):
            if acc[net_idx] > best_acc[net_idx]:
                best_acc[net_idx] = acc[net_idx]
                is_best[net_idx] = 1
        test_acc_list.append(acc)
        test_loss_list.append([test_loss[net_idx] for net_idx in range(num_nets)])

    return best_acc


def train_multi_nodes(num_epochs, nets):
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
            acc = [0] * num_nets

            print('\n=> Training Epoch #%d, LR=[%.4f]'%(epoch+1, current_lr[0]))
            for batch_idx, (original_inputs, targets) in enumerate(trainloader):
                original_inputs, targets = original_inputs.to(device), targets.to(device)  # GPU settings
                original_inputs = preprocess_mnist(original_inputs, block_size=(4, 7))

                for net_idx in range(num_nets):
                    optimizers[net_idx].zero_grad()
                    inputs = original_inputs.clone()
                    if net_idx == 0:
                        inputs = inv_preprocess_mnist(inputs, grid_size=(7, 4))
                        # show_mnist_fig(inputs[0, 0, :, :], "original_image.png")
                    elif net_idx == 1:  # only process high-frequency data
                        inputs[:, :, :, :, :num_spectrals//2] = 0
                        inputs = inv_preprocess_mnist(inputs, grid_size=(7, 4))
                        # show_mnist_fig(inputs[0, 0, :, :], "high_freq_image.png")
                    elif net_idx == 2:  # only process low-frequency data
                        inputs[:, :, :, :, num_spectrals//2:] = 0
                        inputs = inv_preprocess_mnist(inputs, grid_size=(7, 4))
                        # show_mnist_fig(inputs[0, 0, :, :], "low_freq_image.png")

                    outputs = nets[net_idx](inputs)  # Forward Propagation
                    loss[net_idx] = criterion(outputs, targets)  # Loss
                    loss[net_idx].backward()  # Backward Propagation

                    train_loss[net_idx] += loss[net_idx].item()
                    _, predicted = torch.max(outputs.data, 1)
                    total[net_idx] += targets.size(0)
                    correct[net_idx] += predicted.eq(targets.data).cpu().sum().item()
                    acc[net_idx] =  100. * correct[net_idx] / total[net_idx]

                avg_gradients(nets)
                for net_idx in range(num_nets):
                    optimizers[net_idx].step()  # Optimizer update
                sys.stdout.write('\r')
                sys.stdout.write("| Epoch [%3d/%3d] Iter[%3d/%3d]\t"%(epoch+1, num_epochs, batch_idx+1, math.ceil(len(trainset)/batch_size))
                                + ("  Loss: [" + loss_format_str + "]" )%(eval(loss_content_str))
                                + ("  Acc: [" + acc_format_str + "]   ")%(eval(acc_content_str))
                                )
                sys.stdout.flush()
            fusing_weight = [0] * num_nets
            # for i in range(num_nets):
            #     fusing_weight[i] = 1
            # p = 0.3
            # rank_list = np.argsort(train_loss)
            fusing_weight = [0] * num_nets
            for i in range(num_nets):
                # fusing_weight[rank_list[i]] = p * np.power((1 - p), i)
                fusing_weight[i] = 1 / train_loss[i]

            is_best = torch.zeros(size=(num_nets,))
            best_acc = test_multi_nets(epoch, nets, best_acc, test_acc_list, test_loss_list, is_best=is_best)

            train_acc_list.append([100.*correct[i]/total[i] for i in range(num_nets)])
            train_loss_list.append([train_loss[i] / num_train for i in range(num_nets)])
            now_time = time.time()

            print(("| Best Acc: [" + acc_format_str + "]")%(eval(best_acc_content_str)))

            # print("| Best Fusing Acc: [%.2f%%, %.2f%%] "%(best_fusing_acc[0], best_fusing_acc[1]))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))

            if ((epoch + 1) % 10 == 0) or (epoch == 0):
                print("Regularly saving models...")
                name_list = ["low_high_freq", "high_freq", "low_freq"]
                for net_idx in range(num_nets):
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': nets[net_idx].state_dict(),
                            'best_acc1': best_acc[net_idx],
                            "optimizer": optimizers[net_idx].state_dict()
                        },
                        filename="regular_model_{}.pth.tar".format(name_list[net_idx])
                    )

            for net_idx in range(num_nets):
                if is_best[net_idx]:
                    print("Saving best model-{}...".format(net_idx))
                    name_list = ["low_high_freq", "high_freq", "low_freq"]
                    save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'state_dict': nets[net_idx].state_dict(),
                                'best_acc1': best_acc[net_idx],
                                "optimizer": optimizers[net_idx].state_dict()
                            },
                            filename="best_model_{}.pth.tar".format(name_list)
                    )
    except KeyboardInterrupt:
        pass

    print(("\nBest training accuracy overall: [" + acc_format_str + "]")%(eval(best_acc_content_str)))

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, fusing_test_loss_list, fusing_test_acc_list


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc, fusing_test_loss, fusing_test_acc):
    # write csv
    with open('faderated_fc_3_nodes_testloss.csv','w',newline='',encoding='utf-8') as f:
        f_csv = csv.writer(f)

        f_csv.writerow(["Test Acc:"])
        for idx in range(len(test_acc)):
            f_csv.writerow([idx + 1] + test_acc[idx])

        f_csv.writerow(["Test Loss:"])
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + test_loss[idx])

        f_csv.writerow(["Fusing Test Acc:"] + fusing_plan)
        for idx in range(len(fusing_test_acc)):
            f_csv.writerow([idx + 1] + fusing_test_acc[idx])

        f_csv.writerow(["Fusing Test Loss:"] + fusing_plan)
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + fusing_test_loss[idx])
            
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
    plt.title('faderated-fc-3-nodes Loss on MNIST ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_loss[:, i])), test_loss[:, i], label='TestLoss_{}'.format(i+1),linestyle='-')
    for i in range(fusing_num):
        plt.plot(np.arange(len(fusing_test_loss[:, i])), fusing_test_loss[:, i], label='FusingTestLoss_{}'.format(i+1),linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_loss[:, i])), train_loss[:, i], label='TrainLoss_{}'.format(i+1),linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('faderated-fc-3-nodes Accuracy on MNIST ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_acc[:, i])), test_acc[:, i], label='TestAcc_{}'.format(i+1),linestyle='-')
    for i in range(fusing_num):
        plt.plot(np.arange(len(fusing_test_acc[:, i])), fusing_test_acc[:, i], label='FusingTestAcc_{}'.format(i+1),linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_acc[:, i])), train_acc[:, i], label='TrainAcc_{}'.format(i+1),linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./faderated_fc_3_nodes.jpg')


if __name__ == "__main__":
    raw_nets = []
    for _ in range(num_nodes):
        raw_nets.append(build(decomp=False))
    # copy_parameters(raw_nets[1], raw_nets[0])
    # copy_parameters(raw_nets[2], raw_nets[0])
    print(raw_nets[0])
    train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_ = train_multi_nodes(100, raw_nets)
    save_record_and_draw(train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_)
