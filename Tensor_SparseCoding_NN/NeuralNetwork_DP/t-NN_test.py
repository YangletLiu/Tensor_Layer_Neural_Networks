import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def torch_tensor_Bcirc(tensor, l, m, n):
    tensor_blocks = torch.split(tensor, split_size_or_sections=1, dim=0)
    tensor_blocks = list(tensor_blocks)
    tensor_blocks.reverse()

    circ_slices = []
    for i in range(l):
        tensor_blocks.insert(0, tensor_blocks.pop())
        for j in tensor_blocks:
            circ_slices.append(j.reshape(m, n))

    circ = []
    for i in range(l):
        circ.append(torch.cat(circ_slices[l * i:l * i + l], dim=1))
    ulti_circ = torch.cat(circ).reshape(l * m, l * n)

    return ulti_circ


def torch_tensor_product(tensorA, tensorB):
    a_l, a_m, a_n = tensorA.shape
    b_l, b_n, b_p = tensorB.shape

    if a_l == b_l and a_n == b_n:
        circA = torch_tensor_Bcirc(tensorA, a_l, a_m, a_n)
        circB = torch_tensor_Bcirc(tensorB, b_l, b_n, b_p)
        return torch.mm(circA, circB[:, 0:b_p]).reshape(a_l, a_m, b_p)
    else:
        print('Shape Error')


def Softmax_F(out):
    pass


batch_size = 64
lr_rate = 1e-3
epochs_num = 100

# download MNIST
train_datset = datasets.FashionMNIST(
    root='../datasets', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.FashionMNIST(
    root='../datasets', train=False, transform=transforms.ToTensor(), download=False)

# define dataset loader
train_loader = DataLoader(train_datset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class tNN(nn.Module):
    def __init__(self):
        super(tNN, self).__init__()
        """
        use the nn.Parameter() and 'requires_grad = True' 
        to customize parameters which are needed to optimize
        """
        self.W = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
        self.B = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))

    def forward(self, x):
        """
        torch_tensor_product is redefined by torch to complete the tensor-product process
        :param x: x is the input 3D-tensor with shape(l,m,n)
                     'n' denotes the batch_size
        :return: this demo defines an one-layer networks,
                    whose output is processed by one-time tensor-product and activation
        """
        x = torch_tensor_product(self.W, x) + self.B
        x = F.relu(x)
        return x


module = tNN()

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
    print(f'epoch {epoch + 1}')
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0

    module.train()
    for i, data in enumerate(train_loader, 1):
        # load every batch img datasets
        img, label = data
        img = img.view(28, 28, 64)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # forward
        out = module(img)

        # softmax function
        out = Softmax_F(out)
        loss = Loss_function(out, label)
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f'[{epoch + 1}/{epochs_num}] Loss:{running_loss / i:.6f} Acc:{running_acc / i:.6f}')
    print(f'Finish {epoch + 1} epoch, Loss:{running_loss / i:.6f}, Acc:{running_acc / i:.6f}')

    # model evaluate
    module.eval()
    eval_loss = 0.0
    eval_acc = 0.0

    for data in test_loader:
        img, label = data
        img = img.view(28, 28, batch_size)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        with torch.no_grad():
            out = module(img)
            out = Softmax_F(out)
            loss = Loss_function(out, label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()
    print(f'Test Loss: {eval_loss / len(test_loader):.6f}, Acc: {eval_acc / len(test_loader):.6f}')
    print(f'Time:{(time.time() - since):.1f} s')
    loss_after_epoch.append(eval_loss)
    acc_after_epoch.append(eval_acc)

# save the model
torch.save(module.state_dict(), './NeuralNetwork.pth')

fig = plt.figure(figsize=(20, 10))
plt.plot(np.arange(len(loss_after_epoch)), loss_after_epoch, label='Loss')
plt.plot(np.arange(len(acc_after_epoch)), acc_after_epoch, label='Acc')
plt.legend()
plt.show()
