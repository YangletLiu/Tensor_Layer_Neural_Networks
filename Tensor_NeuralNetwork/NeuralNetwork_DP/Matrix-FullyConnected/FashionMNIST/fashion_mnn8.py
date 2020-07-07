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

def mnn_8():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # define super params
    batch_size = 100
    learning_rate = 1e-1
    num_epochs = 100

    # download MNIST
    train_datset = datasets.FashionMNIST(
        root='../datasets', train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        ), download=True
    )

    test_dataset = datasets.FashionMNIST(
        root='../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        ), download=False
    )

    # define dataset loader
    train_loader = DataLoader(train_datset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # define NN model
    class neuralNetwork(nn.Module):

        """in_dem represents the dimension of input
            n_hidden_1,n_hidden_2,n_hidden_3 denotes the three hidden layers' number"""

        def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3,
                     n_hidden_4,n_hidden_5,n_hidden_6,n_hidden_7,out_dim):
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
                nn.ReLU(True)
            )

        # forward
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            return x

    # define NN model
    model = neuralNetwork(28 * 28, 784, 784, 784,784, 784, 784, 784,10)

    # test if GPU is available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    # define loss and optimizer
    # CrossEntropyLoss Function
    criterion = nn.CrossEntropyLoss()
    # SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    test_loss_epoch = []
    test_acc_epoch = []
    train_loss_epoch = []
    train_acc_epoch = []

    # begain train
    for epoch in range(num_epochs):
        print('*' * 10)
        print(f'epoch {epoch + 1}')
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

            if i % 200 == 0:
                print(f'[{epoch + 1}/{num_epochs}] Loss:{running_loss / i:.6f} Acc:{running_acc / i:.6f}')
        print(f'Finish {epoch + 1} epoch, Loss:{running_loss / i:.6f}, Acc:{running_acc / i:.6f}')
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
            with torch.no_grad():
                out = model(img)
                loss = criterion(out, label)
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_acc += (pred == label).float().mean()
        print(f'Test Loss: {eval_loss / j:.6f}, Acc: {eval_acc / j:.6f}')
        print(f'Time:{(time.time() - since):.1f} s')
        test_loss_epoch.append(eval_loss / j)
        test_acc_epoch.append((eval_acc / j) * 100)

    return train_loss_epoch,train_acc_epoch,test_loss_epoch,test_acc_epoch

test_loss,test_acc,train_loss,train_acc = mnn_8()

# write csv
with open('fashion_mnn8_testloss.csv','w',newline='',encoding='utf-8') as f:
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
plt.title('MNN Loss ')
plt.plot(np.arange(len(test_loss)), test_loss, color='red', label='TestLoss',linestyle='-')
plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='TrainLoss',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

sub2 = plt.subplot(1, 2, 2)
plt.sca(sub2)
plt.title('MNN Accuracy ')
plt.plot(np.arange(len(test_acc)), test_acc, color='green', label='TestAcc',linestyle='-')
plt.plot(np.arange(len(train_acc)), train_acc, color='orange', label='TrainAcc',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')

plt.legend()
plt.show()

plt.savefig('./fashion_mnn8.jpg')
