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
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010]),
                                ])

transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
                                ])

num_nets = 4
batch_size = 128
trainset = datasets.CIFAR10(root='../datasets', train=True, transform=transform_train, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
num_train = len(trainset)

testset = datasets.CIFAR10(root='../datasets', train=False, transform=transform_test, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
num_test = len(testset)


########################### 2. define model ##################################
class CNN8CIFAR10(nn.Module):
    def __init__(self):
        super(CNN8CIFAR10,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pred = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4*32*32,10)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.reshape(x.size(0),-1)
        x = self.pred(x)
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


def preprocess_cifar(x):
    permute_plan = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 3, 1, 2], [3, 0, 1, 2]]
    x = x.permute([0, 2, 3, 1])
    # generate the 'average channel', and attach the 'average channel' to the original 3 channels
    thicker_x = torch.cat([x, torch.mean(x, dim=-1, keepdim=True)], dim=-1)
    first_channels = []
    second_channels = []
    third_channels = []
    fourth_channels = []
    for i in range(len(permute_plan)):
        rt = dct(thicker_x[:, :, :, permute_plan[i]])
        first_channels.append(rt[:, :, :, 0])
        second_channels.append(rt[:, :, :, 1])
        third_channels.append(rt[:, :, :, 2])
        fourth_channels.append(rt[:, :, :, 3])

    new_x = []
    first_channels = torch.stack(first_channels, dim=-1)
    first_channels = first_channels.permute([0, 3, 1, 2]).contiguous()
    new_x.append(first_channels)

    second_channels = torch.stack(second_channels, dim=-1)
    second_channels = second_channels.permute([0, 3, 1, 2]).contiguous()
    new_x.append(second_channels)

    third_channels = torch.stack(third_channels, dim=-1)
    third_channels = third_channels.permute([0, 3, 1, 2]).contiguous()
    new_x.append(third_channels)

    fourth_channels = torch.stack(fourth_channels, dim=-1)
    fourth_channels = fourth_channels.permute([0, 3, 1, 2]).contiguous()
    new_x.append(fourth_channels)
    return new_x

# build model
def build(decomp=False):
    print('==> Building model..')
    full_net = CNN8CIFAR10()
    if decomp:
        raise("No Tensor Neural Network decompostion implementation.")
    print('==> Done')
    return full_net


########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss().to(device)
lr0 = [0.001, 0.001, 0.001, 0.001]
fusing_plan = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
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
            img = preprocess_cifar(img)

            for i in range(num_nets):
                outputs[i] = nets[i](img[i])
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
              %(epoch+1, test_loss[0], test_loss[1], test_loss[2], test_loss[3], 
                acc[0], acc[1], acc[2], acc[3]))

        print("| Fusing Loss: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n "
              %(fusing_test_loss[0], fusing_test_loss[1], fusing_test_loss[2], fusing_test_loss[3], fusing_test_loss[4], fusing_test_loss[5],
                fusing_test_loss[6], fusing_test_loss[7], fusing_test_loss[8], fusing_test_loss[9], fusing_test_loss[10])
              +"Fusing Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%]   " 
              %(fusing_acc[0], fusing_acc[1], fusing_acc[2], fusing_acc[3], fusing_acc[4], fusing_acc[5], 
                fusing_acc[6], fusing_acc[7], fusing_acc[8], fusing_acc[9], fusing_acc[10]))

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
            img = preprocess_cifar(img)

            for i in range(num_nets):

                outputs = nets[i](img[i])
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
        print("\n| Validation Epoch #%d\t\tLoss: [%.4f, %.4f, %.4f, %.4f] Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%]   " 
              %(epoch+1, test_loss[0], test_loss[1], test_loss[2], test_loss[3],
                acc[0], acc[1], acc[2], acc[3]))

        for i in range(num_nets):
            if acc[i] > best_acc[i]:
                best_acc[i] = acc[i]
                torch.save(nets[i], "./4_channel_fusing_cnn_8_channel_{}.pth".format(i))
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
        optimizers.append(torch.optim.SGD(nets[i].parameters(), lr=lr0[i], momentum=0.9))
        # optimizers.append(torch.optim.Adam(nets[i].parameters(), lr=lr0[i]))

    current_lr = lr0

    try:
        for epoch in range(num_epochs):
            for net in nets:
                net.train()
            train_loss = [0] * num_nets
            correct = [0] * num_nets
            total = [0] * num_nets
            loss = [0] * num_nets

            print('\n=> Training Epoch #%d, LR=[%.4f, %.4f, %.4f, %.4f]' %(epoch+1, current_lr[0], current_lr[1], current_lr[2], current_lr[3]))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                inputs = preprocess_cifar(inputs)
                for i in range(num_nets):
                    optimizers[i].zero_grad()
                    outputs = nets[i](inputs[i])               # Forward Propagation
                    loss[i] = criterion(outputs, targets)  # Loss
                    loss[i].backward()  # Backward Propagation
                    optimizers[i].step() # Optimizer update

                    train_loss[i] += loss[i].item()
                    _, predicted = torch.max(outputs.data, 1)
                    total[i] += targets.size(0)
                    correct[i] += predicted.eq(targets.data).cpu().sum().item()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: [%.4f, %.4f, %.4f, %.4f] Acc: [%.3f%%, %.3f%%, %.3f%%, %.3f%%]   '
                        %(epoch+1, num_epochs, batch_idx+1, math.ceil(len(trainset)/batch_size), 
                          loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(),
                          100.*correct[0]/total[0], 100.*correct[1]/total[1], 100.*correct[2]/total[2], 100.*correct[3]/total[3]))
                sys.stdout.flush()

            fusing_weight = [0] * num_nets
            for i in range(num_nets):
                fusing_weight[i] = 1 / train_loss[i]

            best_acc, best_fusing_acc = test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list,
                                        fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_weight=fusing_weight)

            train_acc_list.append([100.*correct[i]/total[i] for i in range(num_nets)])
            train_loss_list.append([train_loss[i] / num_train for i in range(num_nets)])
            now_time = time.time()
            print("| Best Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%] "%(best_acc[0], best_acc[1], best_acc[2], best_acc[3]))
            print("| Best Fusing Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%] "%(
                    best_fusing_acc[0], best_fusing_acc[1], best_fusing_acc[2], best_fusing_acc[3],
                    best_fusing_acc[4], best_fusing_acc[5], best_fusing_acc[6], best_fusing_acc[7],
                    best_fusing_acc[8], best_fusing_acc[9], best_fusing_acc[10]))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    print("\nBest training accuracy overall: [%.3f%%, %.3f%%, %.3f%%, %.3f%%] "%(best_acc[0], best_acc[1], best_acc[2], best_acc[3]))
    print("| Best Fusing Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%, %.2f%%] "%(
            best_fusing_acc[0], best_fusing_acc[1], best_fusing_acc[2], best_fusing_acc[3],
            best_fusing_acc[4], best_fusing_acc[5], best_fusing_acc[6], best_fusing_acc[7],
            best_fusing_acc[8], best_fusing_acc[9], best_fusing_acc[10]))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, fusing_test_loss_list, fusing_test_acc_list


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc, fusing_test_loss, fusing_test_acc):

    # write csv
    with open('4_channel_fusing_cnn_8_cifar10_testloss.csv','w',newline='',encoding='utf-8') as f:
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
    plt.title('4-channel fusing-cnn-8 Loss on CIFAR10 ')
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
    plt.title('4-channel fusing-cnn-8 Accuracy on CIFAR10 ')
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

    plt.savefig('./4_channel_fusing_cnn_8_cifar10.jpg')


if __name__ == "__main__":
    raw_nets = []
    for _ in range(num_nets):
        raw_nets.append(build(decomp=False))
    print(raw_nets[0])
    train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_ = train_multi_nets(300, raw_nets)
    save_record_and_draw(train_loss_, train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_)
