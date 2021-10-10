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

num_nets = 28
batch_size = 100
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


def preprocess_mnist(x):
    # x = dct(x)
    # return x
    pass

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


def query_lr(epoch):
    lr = lr0
    return lr


def set_lr(optimizer, epoch):
    current_lr = query_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list, fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_weight=None):
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
            img = dct(img)

            for i in range(num_nets):
                outputs[i] = nets[i](img[:, :, :, i])
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

        print("\n| Validation Epoch #%d\t\t"%(epoch+1)
              +"Loss: [%.4f, %.4f, %.4f, %.4f, "%(test_loss[0], test_loss[1], test_loss[2], test_loss[3])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[4], test_loss[5], test_loss[6], test_loss[7])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[8], test_loss[9], test_loss[10], test_loss[11])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[12], test_loss[13], test_loss[14], test_loss[15])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[16], test_loss[17], test_loss[18], test_loss[19])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[20], test_loss[21], test_loss[22], test_loss[23])
              +"%.4f, %.4f, %.4f, %.4f]"%(test_loss[24], test_loss[25], test_loss[26], test_loss[27])
              +" Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[0], acc[1], acc[2], acc[3])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[4], acc[5], acc[6], acc[7])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[8], acc[9], acc[10], acc[11])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[12], acc[13], acc[14], acc[15])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[16], acc[17], acc[18], acc[19])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[20], acc[21], acc[22], acc[23])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%]"%(acc[24], acc[25], acc[26], acc[27])
            )

        print("| Fusing Loss: [%.4f]\t"%(fusing_test_loss[0])
              +"Fusing Acc: [%.2f%%]  "%(fusing_acc[0]))

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


def test_multi_nets(epoch, nets, best_acc, test_acc_list, test_loss_list):
    for net in nets:
        net.eval()
    test_loss = [0] * num_nets
    correct = [0] * num_nets
    total = [0] * num_nets

    with torch.no_grad():
        for batch_idx, (img, targets) in enumerate(testloader):
            img, targets = img.to(device), targets.to(device)
            img = dct(img)

            for i in range(num_nets):
                outputs = nets[i](img[:, :, :, i])
                loss = criterion(outputs, targets)

                test_loss[i] += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets.data).cpu().sum().item()

        # Save checkpoint when best model
        acc = [0] * num_nets
        for i in range(num_nets):
            test_loss[i] /= num_test
            acc[i] = 100. * correct[i] / total[i]
        print("\n| Validation Epoch #%d\t\t"%(epoch+1)
              +"Loss: [%.4f, %.4f, %.4f, %.4f, "%(test_loss[0], test_loss[1], test_loss[2], test_loss[3])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[4], test_loss[5], test_loss[6], test_loss[7])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[8], test_loss[9], test_loss[10], test_loss[11])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[12], test_loss[13], test_loss[14], test_loss[15])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[16], test_loss[17], test_loss[18], test_loss[19])
              +"%.4f, %.4f, %.4f, %.4f, "%(test_loss[20], test_loss[21], test_loss[22], test_loss[23])
              +"%.4f, %.4f, %.4f, %.4f]"%(test_loss[24], test_loss[25], test_loss[26], test_loss[27])
              +" Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[0], acc[1], acc[2], acc[3])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[4], acc[5], acc[6], acc[7])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[8], acc[9], acc[10], acc[11])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[12], acc[13], acc[14], acc[15])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[16], acc[17], acc[18], acc[19])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(acc[20], acc[21], acc[22], acc[23])
              +"%.2f%%, %.2f%%, %.2f%%, %.2f%%]"%(acc[24], acc[25], acc[26], acc[27])
            )

        for i in range(num_nets):
            if acc[i] > best_acc[i]:
                best_acc[i] = acc[i]
                torch.save(nets[i], "./fusing_tnn_4_mnist_{}.pth".format(i))
        test_acc_list.append(acc)
        test_loss_list.append([test_loss[i] / num_test for i in range(num_nets)])
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

            print('\n=> Training Epoch #%d, LR=[%.4f, %.4f, %.4f, %.4f, ...]' %(epoch+1, current_lr[0], current_lr[1], current_lr[2], current_lr[3]))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                inputs = dct(inputs)
                for i in range(num_nets):
                    optimizers[i].zero_grad()
                    outputs = nets[i](inputs[:, :, :, i])               # Forward Propagation
                    loss[i] = criterion(outputs, targets)  # Loss
                    loss[i].backward()  # Backward Propagation
                    optimizers[i].step() # Optimizer update

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
            for i in range(num_nets):
                fusing_weight[i] = 1 / train_loss[i]

            best_acc, best_fusing_acc = test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list,
                                        fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_weight=fusing_weight)

            train_acc_list.append([100.*correct[i]/total[i] for i in range(num_nets)])
            train_loss_list.append([train_loss[i] / num_train for i in range(num_nets)])
            now_time = time.time()

            print("| Best Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[0], best_acc[1], best_acc[2], best_acc[3])
                    +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[4], best_acc[5], best_acc[6], best_acc[7])
                    +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[8], best_acc[9], best_acc[10], best_acc[11])
                    +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[12], best_acc[13], best_acc[14], best_acc[15])
                    +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[16], best_acc[17], best_acc[18], best_acc[19])
                    +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[20], best_acc[21], best_acc[22], best_acc[23])
                    +"%.2f%%, %.2f%%, %.2f%%, %.2f%%]"%(best_acc[24], best_acc[25], best_acc[26], best_acc[27])
                )
            print("| Best Fusing Acc: [%.2f%%] "%(best_fusing_acc[0]))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    print("\nBest training accuracy overall: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[0], best_acc[1], best_acc[2], best_acc[3])
            +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[4], best_acc[5], best_acc[6], best_acc[7])
            +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[8], best_acc[9], best_acc[10], best_acc[11])
            +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[12], best_acc[13], best_acc[14], best_acc[15])
            +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[16], best_acc[17], best_acc[18], best_acc[19])
            +"%.2f%%, %.2f%%, %.2f%%, %.2f%%, "%(best_acc[20], best_acc[21], best_acc[22], best_acc[23])
            +"%.2f%%, %.2f%%, %.2f%%, %.2f%%]"%(best_acc[24], best_acc[25], best_acc[26], best_acc[27])
        )
    print("| Best Fusing Acc: [%.2f%%] "%(best_fusing_acc[0]))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, fusing_test_loss_list, fusing_test_acc_list


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc, fusing_test_loss, fusing_test_acc):

    # write csv
    with open('8_layer_spectral_tensor_mnist_testloss.csv','w',newline='',encoding='utf-8') as f:
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
    plt.title('8-layer-spectral-tensor Loss on MNIST ')
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
    plt.title('8-layer-spectral-tensor Accuracy on MNIST ')
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

    plt.savefig('./8_layer_spectral_tensor_mnist.jpg')


if __name__ == "__main__":
    raw_nets = []
    for _ in range(num_nets):
        raw_nets.append(build(decomp=False))
    print(raw_nets[0])
    train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_ = train_multi_nets(300, raw_nets)
    save_record_and_draw(train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_)
