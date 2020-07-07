# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv

def tnn_4():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # dct and idct
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

        Vc = torch.rfft(v, 1, onesided=False)

        k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)).cuda()
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

        k = (torch.arange(x_shape[-1], dtype=X.dtype)[None, :] * np.pi / (2 * N)).cuda()
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.irfft(V, 1, onesided=False)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    # circulant convolution part
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

    # Loss function(scalar tubal softmax function)
    def h_func_dct(lateral_slice):
        l, m, n = lateral_slice.shape

        dct_slice = dct(lateral_slice)

        tubes = [dct_slice[i, :, 0] for i in range(l)]

        h_tubes = []
        for tube in tubes:
            tube_sum = torch.sum(torch.exp(tube))
            h_tubes.append(torch.exp(tube) / tube_sum)

        res_slice = torch.stack(h_tubes, dim=0).reshape(l, m, n)

        idct_a = idct(res_slice)

        return torch.sum(idct_a, dim=0)

    def scalar_tubal_func(output_tensor):
        l, m, n = output_tensor.shape

        lateral_slices = [output_tensor[:, :, i].reshape(l, m, 1) for i in range(n)]

        h_slice = []
        for slice in lateral_slices:
            h_slice.append(h_func_dct(slice))

        pro_matrix = torch.stack(h_slice, dim=2)
        return pro_matrix.reshape(m, n)

    # process raw
    def raw_img(img, batch_size, n):
        img_raw = img.reshape(batch_size, n * n)
        single_img = torch.split(img_raw, split_size_or_sections=1, dim=0)
        single_img_T = [torch.transpose(i.reshape(n, n, 1), 0, 1) for i in single_img]
        ultra_img = torch.cat(single_img_T, dim=2)
        return ultra_img

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

    # define nn module
    class tNN(nn.Module):
        def __init__(self):
            super(tNN, self).__init__()
            """
            use the nn.Parameter() and 'requires_grad = True' 
            to customize parameters which are needed to optimize
            """
            self.W_1 = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
            self.B_1 = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))
            self.W_2 = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
            self.B_2 = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))
            self.W_3 = nn.Parameter(torch.randn((28, 28, 28), requires_grad=True, dtype=torch.float))
            self.B_3 = nn.Parameter(torch.randn((28, 28, 1), requires_grad=True, dtype=torch.float))
            self.W_4 = nn.Parameter(torch.randn((28, 10, 28), requires_grad=True, dtype=torch.float))
            self.B_4 = nn.Parameter(torch.randn((28, 10, 1), requires_grad=True, dtype=torch.float))

        def forward(self, x):
            """
            torch_tensor_product is redefined by torch to complete the tensor-product process
            :param x: x is the input 3D-tensor with shape(l,m,n)
                         'n' denotes the batch_size
            :return: this demo defines an one-layer networks,
                        whose output is processed by one-time tensor-product and activation
            """
            x = torch_tensor_product(self.W_1, x) + self.B_1
            x = F.relu(x)
            x = torch_tensor_product(self.W_2, x) + self.B_2
            x = F.relu(x)
            x = torch_tensor_product(self.W_3, x) + self.B_3
            x = F.relu(x)
            x = torch_tensor_product(self.W_4, x) + self.B_4
            x = F.relu(x)
            return x

    module = tNN()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        module = module.cuda()

    Loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=lr_rate)

    test_loss_epoch = []
    test_acc_epoch = []
    train_loss_epoch = []
    train_acc_epoch = []

    # begain train
    for epoch in range(epochs_num):
        print('*' * 10)
        print('epoch {Epoch}'.format(Epoch=epoch + 1))
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0

        module.train()
        for i, data in enumerate(train_loader, 1):
            img, label = data
            img = raw_img(img, img.size(0), 28)

            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            # forward
            out = module(img) / 1e5
            # print(out)

            # softmax function
            out = torch.transpose(scalar_tubal_func(out), 0, 1)
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
                      .format(Epoch=epoch + 1, Epochs_num=epochs_num, Running_loss=(running_loss / i),
                              Running_acc=running_acc / i))
        print('[{Epoch}/{Epochs_num}] Loss:{Running_loss} Acc:{Running_acc}'
              .format(Epoch=epoch + 1, Epochs_num=epochs_num, Running_loss=(running_loss / i),
                      Running_acc=running_acc / i))
        train_loss_epoch.append(running_loss/i)
        train_acc_epoch.append((running_acc/i)*100)

        module.eval()
        eval_loss = 0.0
        eval_acc = 0.0

        for data in test_loader:
            img, label = data
            img = raw_img(img, img.size(0), 28)

            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            with torch.no_grad():
                out = module(img) / 1e5
                out = torch.transpose(scalar_tubal_func(out), 0, 1)
                loss = Loss_function(out, label)
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_acc += (pred == label).float().mean()
        print('Test Loss: {Eval_loss}, Acc: {Eval_acc}'
              .format(Eval_loss=eval_loss / len(test_loader), Eval_acc=eval_acc / len(test_loader)))
        print('Time:{Time} s'.format(Time=time.time() - since))
        test_loss_epoch.append(eval_loss / len(test_loader))
        test_acc_epoch.append((eval_acc / len(test_loader)) * 100)

        if np.isnan(eval_loss):
            break

    return test_loss_epoch,test_acc_epoch,train_loss_epoch,train_acc_epoch

# save the model
# torch.save(module.state_dict(), './NeuralNetwork—8.pth')

test_loss,test_acc,train_loss,train_acc = tnn_4()

# write csv
with open('tnn4_testloss.csv','w',newline='',encoding='utf-8') as f:
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
plt.title('TNN Loss ')
plt.plot(np.arange(len(test_loss)), test_loss, color='red', label='TestLoss',linestyle='-')
plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='TrainLoss',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

sub2 = plt.subplot(1, 2, 2)
plt.sca(sub2)
plt.title('TNN Accuracy ')
plt.plot(np.arange(len(test_acc)), test_acc, color='green', label='TestAcc',linestyle='-')
plt.plot(np.arange(len(train_acc)), train_acc, color='orange', label='TrainAcc',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')

plt.legend()
plt.show()

plt.savefig('./tnn4.jpg')
