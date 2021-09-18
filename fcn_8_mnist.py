######################### 0. import packages #############################
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
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

batch_size = 64
trainset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
num_train = len(trainset)

testset = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
num_test = len(testset)

########################### 2. define model ##################################
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
def low_rank_matrix_decompose_FC_layer(layer, r):
    print("'low rank matrix' decompose one FC layer")
    # y = Wx + b ==>  y = W1(W2x) + b
    lj, lj_1 = layer.weight.data.shape

    D = nn.Linear(in_features=lj_1,
                   out_features=r,
                   bias=False)
    C = nn.Linear(in_features=r,
                   out_features=lj,
                   bias=True)

    C.weight.data = torch.randn(lj, r) * torch.sqrt(torch.tensor(2 / (lj + r)))
    D.weight.data = torch.randn(r, lj_1) * torch.sqrt(torch.tensor(2 / (r + lj_1)))
    C.bias.data = torch.randn(lj, ) * torch.sqrt(torch.tensor(2 / (lj + 1)))

    new_layers = [D, C]
    return nn.Sequential(*new_layers)


def low_rank_matrix_decompose_nested_FC_layer(layer):
    modules = layer._modules
    for key in modules.keys():
        l = modules[key]
        if isinstance(l, nn.Sequential):
            modules[key] = low_rank_matrix_decompose_nested_FC_layer(l)
        elif isinstance(l, nn.Linear):
            fc_layer = l
            sp = fc_layer.weight.data.numpy().shape
            # rank = min(max(sp)//8, min(sp))
            rank = 8
            modules[key] = low_rank_matrix_decompose_FC_layer(fc_layer, rank)
    return layer


# decomposition
def decompose_FC(model, mode):
    model.eval()
    model.cpu()
    layers = model._modules # model.features._modules  # model._modules
    cnt = 1
    for i, key in enumerate(layers.keys()):
        if isinstance(layers[key], torch.nn.modules.Linear):
            fc_layer = layers[key]
            sp = fc_layer.weight.data.numpy().shape
            rank = 8
            if rank == min(sp):
                continue
            if mode == "low_rank_matrix":
                layers[key] = low_rank_matrix_decompose_FC_layer(fc_layer, rank)
        elif isinstance(layers[key], nn.Sequential):
            if mode == "low_rank_matrix":
                layers[key] = low_rank_matrix_decompose_nested_FC_layer(layers[key])
    return model


# build model
def build(decomp=True):
    print('==> Building model..')
    full_net = FC8Net(784, 784, 784, 784, 784, 784, 784, 784, 10)
    if decomp:
        full_net = decompose_FC(full_net, mode="low_rank_matrix")
    full_net = full_net.to(device)
    print('==> Done')
    return full_net


########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss().to(device)
lr0 = 0.01


def query_lr(epoch):
    lr = lr0
    return lr


def set_lr(optimizer, epoch):
    current_lr = query_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def test(epoch, net, best_acc, test_acc_list, test_loss_list):
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.* correct / total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc: %.2f%%   " %(epoch, loss.item(), acc))

        if acc > best_acc:
            best_acc = acc.item()
        test_acc_list.append(acc)
        test_loss_list.append(test_loss / num_test)
    return best_acc


# Training
def train(num_epochs, net):
    net = net.to(device)
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = 0.

    original_time = time.asctime(time.localtime(time.time()))
    start_time = time.time()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr0)
    current_lr = lr0

    try:
        for epoch in range(num_epochs):
            net.train()
            net.training = True
            train_loss = 0
            correct = 0
            total = 0

            current_lr = set_lr(optimizer, epoch)
            print('\n=> Training Epoch #%d, LR=%.4f' %(epoch+1, current_lr))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                optimizer.zero_grad()
                outputs = net(inputs)               # Forward Propagation
                loss = criterion(outputs, targets)  # Loss
                loss.backward()  # Backward Propagation
                optimizer.step() # Optimizer update

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%   '
                        %(epoch+1, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
                sys.stdout.flush()

            best_acc = test(epoch, net, best_acc, test_acc_list, test_loss_list)
            train_acc_list.append(100.*correct/total)
            train_loss_list.append(train_loss / num_train)
            now_time = time.time()
            print("| Best Acc: %.2f%% "%(best_acc))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    print("\nBest training accuracy overall: %.3f%%"%(best_acc))

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


def save_record_and_draw(train_loss, train_acc, test_loss, test_acc):
    # write csv
    with open('fcn_8_mnist_testloss.csv','w',newline='',encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['Test Loss:'])
        f_csv.writerows(enumerate(test_loss,1))
        f_csv.writerow(['Train Loss:'])
        f_csv.writerows(enumerate(train_loss,1))
        f_csv.writerow(['Test Acc:'])
        f_csv.writerows(enumerate(test_acc,1))
        f_csv.writerow(['Train Acc:'])
        f_csv.writerows(enumerate(train_acc,1))

    # draw picture
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title('FCN-8 Loss on MNIST ')
    plt.plot(np.arange(len(test_loss)), test_loss, color='red', label='TestLoss',linestyle='-')
    plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='TrainLoss',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('FCN-8 Accuracy on MNIST ')
    plt.plot(np.arange(len(test_acc)), test_acc, color='green', label='TestAcc',linestyle='-')
    plt.plot(np.arange(len(train_acc)), train_acc, color='orange', label='TrainAcc',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./fcn_8_mnist.jpg')


if __name__ == "__main__":
    net = build(decomp=False)
    print(net)
    train_loss, train_acc, test_loss, test_acc = train(100, net)
    save_record_and_draw(train_loss, train_acc, test_loss, test_acc)
