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

batch_size = 128
trainset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
num_train = len(trainset)

testset = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
num_test = len(testset)

########################### 2. define model ##################################
class CNN4MNIST(nn.Module):
    def __init__(self):
        super(CNN4MNIST,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.pred = nn.Linear(32*3*3,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x

######################## 3. build model functions #################

# build model
def build(decomp=False):
    print('==> Building model..')
    full_net = CNN4MNIST()
    # print(full_net)
    # full_net.apply(weight_init)
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
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc: %.2f%%   " %(epoch+1, loss.item(), acc))

        if acc > best_acc:
            best_acc = acc
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

            # scheduler.step()
            # current_lr = scheduler.get_last_lr()[0]  # query_lr(epoch)
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
    with open('cnn_4_mnist_testloss.csv','w',newline='',encoding='utf-8') as f:
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
    plt.title('CNN-4 Loss on MNIST ')
    plt.plot(np.arange(len(test_loss)), test_loss, color='red', label='TestLoss',linestyle='-')
    plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='TrainLoss',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('CNN-4 Accuracy on MNIST ')
    plt.plot(np.arange(len(test_acc)), test_acc, color='green', label='TestAcc',linestyle='-')
    plt.plot(np.arange(len(train_acc)), train_acc, color='orange', label='TrainAcc',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./cnn_4_mnist.jpg')


if __name__ == "__main__":
    net = build(decomp=False)
    print(net)
    train_loss, train_acc, test_loss, test_acc = train(100, net)
    save_record_and_draw(train_loss, train_acc, test_loss, test_acc)
