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


def fcn_4():
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # define super params
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 100

    # download MNIST
    train_datset = datasets.CIFAR10(
        root='../datasets', train=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]
             )
             ]
        ), download=True)

    test_dataset = datasets.CIFAR10(
        root='../datasets', train=False, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]
             )
             ]
        ), download=True)

    # define dataset loader
    train_loader = DataLoader(train_datset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    # define NN model
    class neuralNetwork(nn.Module):

        """in_dem represents the dimension of input
            n_hidden_1,n_hidden_2,n_hidden_3 denotes the three hidden layers' number"""

        def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
            super(neuralNetwork, self).__init__()
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
                nn.Linear(n_hidden_3, out_dim),
            )

        # forward
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

    # define NN model
    model = neuralNetwork(3072, 4096, 2048, 1024, 10)
    model.apply(weight_init)

    # test if GPU is available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    # define loss and optimizer
    # CrossEntropyLoss Function
    criterion = nn.CrossEntropyLoss()
    # SGD
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_loss_epoch = []
    test_acc_epoch = []
    train_loss_epoch = []
    train_acc_epoch = []

    # begain train
    for epoch in range(num_epochs):
        print('\n=> Training Epoch #%d' %(epoch+1))
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0

        model.train()
        for i, data in enumerate(train_loader, 1):
            # load every single train sample img
            img, label = data
            # print(img.shape)
            img = img.view(img.size(0), -1)
            # print(img.size(0))


            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            # forward
            out = model(img)
            # print(out.shape)
            loss = criterion(out, label)

            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            # print(pred)
            running_acc += (pred == label).float().mean()
            # print(running_acc)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%    '
                    %(epoch, num_epochs, i,
                        (len(train_datset)//batch_size)+1, loss.item(), 100.* running_acc / i))
            sys.stdout.flush()

        print(f'\nFinish {epoch + 1} epoch, Avg Loss:{running_loss / i:.6f}, Acc:{100. * running_acc / i:.6f}%')
        train_loss_epoch.append(running_loss / i)
        train_acc_epoch.append((running_acc / i)*100)

        # model evaluate
        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0

        for j, data in enumerate(test_loader, 1):
            img, label = data
            img = img.view(img.size(0), -1)
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_acc += (pred == label).float().mean()
        print(f'Test Loss: {eval_loss / j:.6f}, Acc: {100. * eval_acc / j:.6f}%')
        print(f'Time:{(time.time() - since):.1f} s')
        test_loss_epoch.append(eval_loss / j)
        test_acc_epoch.append((eval_acc / j) * 100)

    return train_loss_epoch,train_acc_epoch,test_loss_epoch,test_acc_epoch

train_loss,train_acc,test_loss,test_acc = fcn_4()

# write csv
with open('cifar10_fcn_4_testloss.csv','w',newline='',encoding='utf-8') as f:
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
plt.title('FCN-4 Loss on CIFAR10 ')
plt.plot(np.arange(len(test_loss)), test_loss, color='red', label='TestLoss',linestyle='-')
plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='TrainLoss',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

sub2 = plt.subplot(1, 2, 2)
plt.sca(sub2)
plt.title('FCN-4 Accuracy on CIFAR10')
plt.plot(np.arange(len(test_acc)), test_acc, color='green', label='TestAcc',linestyle='-')
plt.plot(np.arange(len(train_acc)), train_acc, color='orange', label='TrainAcc',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')

plt.legend()
plt.show()

plt.savefig('./cifar10_fcn_4.jpg')
