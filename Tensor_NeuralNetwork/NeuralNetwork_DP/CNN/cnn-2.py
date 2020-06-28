import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# define super parameters
batch_size = 100
lr_rate = 0.1
epochs_num = 100

# download MNIST
train_datset = datasets.MNIST(
    root='../datasets', train=True, transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    ), download=True)

test_dataset = datasets.MNIST(
    root='../datasets', train=False, transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    ), download=True)

# define dataset loader
train_loader = DataLoader(train_datset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.pred = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x

module = CNN()
use_gpu = torch.cuda.is_available()
if use_gpu:
    module = module.cuda()

Loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(module.parameters(), lr=lr_rate)

loss_after_epoch = []
acc_after_epoch = []

# begain train
for epoch in range(epochs_num):
    print('*' * 10)
    print('epoch {Epoch}'.format(Epoch=epoch+1))
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0

    loss_epoch = []
    module.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # forward
        out = module(img)
        # print(out)

        # softmax function
        loss = Loss_function(out, label)
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print('[{Epoch}/{Epochs_num}] Loss:{Running_loss} Acc:{Running_acc}'
                  .format(Epoch=epoch+1,Epochs_num=epochs_num,Running_loss=(running_loss/i),Running_acc=running_acc / i))
    print('[{Epoch}/{Epochs_num}] Loss:{Running_loss} Acc:{Running_acc}'
          .format(Epoch=epoch + 1, Epochs_num=epochs_num, Running_loss=(running_loss / i), Running_acc=running_acc / i))
    loss_epoch.append(running_loss)

    module.eval()
    eval_loss = 0.0
    eval_acc = 0.0

    for data in test_loader:
        img, label = data

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        with torch.no_grad():
            out = module(img)
            loss = Loss_function(out, label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()
    print('Test Loss: {Eval_loss}, Acc: {Eval_acc}'
          .format(Eval_loss=eval_loss / len(test_loader),Eval_acc=eval_acc / len(test_loader)))
    print('Time:{Time} s'.format(Time=time.time()-since))
    loss_after_epoch.append(eval_loss / len(test_loader))
    acc_after_epoch.append((eval_acc / len(test_loader)) * 100)

    if np.isnan(eval_loss):
        break

# save the model
torch.save(module.state_dict(), './Convd-NeuralNetwork.pth')

# draw picture
fig = plt.figure(1)
sub1 = plt.subplot(1, 2, 1)
plt.sca(sub1)
plt.plot(np.arange(len(loss_after_epoch)), loss_after_epoch, color='red', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

sub2 = plt.subplot(1, 2, 2)
plt.sca(sub2)
plt.plot(np.arange(len(acc_after_epoch)), acc_after_epoch, color='blue', label='Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')

plt.legend()
plt.show()

plt.savefig('./tnn_test3.jpg')
