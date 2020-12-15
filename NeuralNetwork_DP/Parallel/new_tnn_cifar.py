import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import pkbar
import math
from torchvision import models
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('../')
#from common import *
#from transform_based_network import *

def load_mnist_multiprocess(override=0):
    print('==> Loading data..')
    if override:
        cpu_count = override
    else:
        cpu_count = os.cpu_count()
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cpu_count, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cpu_count, shuffle=False, num_workers=0)
    return trainloader, testloader
    
def load_cifar10():
    print('==> Loading data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)   
    return trainloader, testloader   

def raw_img(img, batch_size, n):
    img_raw = img.reshape(batch_size, n * n)
    single_img = torch.split(img_raw, split_size_or_sections=1, dim=0)
    single_img_T = [torch.transpose(i.reshape(n, n, 1), 0, 1) for i in single_img]
    ultra_img = torch.cat(single_img_T, dim=2)
    return ultra_img

def cifar_img_process(raw_img):
    k, l, m, n = raw_img.shape
    img_list = torch.split(raw_img, split_size_or_sections=1, dim=0)
    list = []
    for img in img_list:
        img = img.reshape(l, m, n)
        frontal = torch.cat([img[i, :, :] for i in range(l)], dim=0)
        single_img = torch.transpose(frontal.reshape(1, l * m, n), 0, 2)
        list.append(single_img)
    ultra_img = torch.cat(list, dim=2)
    return ultra_img
    
def cifar_img_processR(raw_img):
    k, l, m, n = raw_img.shape
    img_list = torch.split(raw_img, split_size_or_sections=1, dim=0)
    list=[]
    for img in img_list:
        img = img.reshape(l, m, n)
        list.append(img[0, :, :])
    ultra_img = torch.cat(list,0)
    ultra_img = ultra_img.reshape(k,m,n)
    return ultra_img
    
def torch_apply(func, x):
    x = func(torch.transpose(x, 0, 2))
    return torch.transpose(x, 0, 2)
    
def h_func_dct(lateral_slice):
    l, m, n = lateral_slice.shape
    dct_slice = dct.dct(lateral_slice)
    tubes = [dct_slice[i, :, 0] for i in range(l)]
    h_tubes = []
    for tube in tubes:
        h_tubes.append(torch.exp(tube) / torch.sum(torch.exp(tube)))
    res_slice = torch.stack(h_tubes, dim=0).reshape(l, m, n)
    idct_a = res_slice#dct.idct(res_slice)
    return torch.sum(idct_a, dim=0) 
    
def scalar_tubal_func(output_tensor):
    l, m, n = output_tensor.shape
    lateral_slices = [output_tensor[:, :, i].reshape(l, m, 1) for i in range(n)]
    h_slice = []
    for s in lateral_slices:
        h_slice.append(h_func_dct(s))
    pro_matrix = torch.stack(h_slice, dim=2)
    return pro_matrix.reshape(m, n)
    
def t_product_in_network(A, B):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    dct_C = torch.zeros(A.shape[0], A.shape[1], B.shape[2])
    dct_A = torch_apply(dct.dct, A)
    for k in range(A.shape[0]):
        dct_C[k, ...] = torch.mm(dct_A[k, ...], B[k, ...])
    return dct_C #.to(device)
        
class new_tNN(nn.Module):
    def __init__(self):
        super(new_tNN, self).__init__()
        W, B = [], []
        self.num_layers = 4
        for i in range(self.num_layers):
            W.append(nn.Parameter(torch.Tensor(32, 32, 32)))
            B.append(nn.Parameter(torch.Tensor(32, 32, 1)))
        self.W = nn.ParameterList(W)
        self.B = nn.ParameterList(B)
        self.reset_parameters()

    def forward(self, x):
        x = torch_apply(dct.dct, x)
        for i in range(self.num_layers):
            x = torch.add(t_product_in_network(self.W[i], x), self.B[i])
            x = F.relu(x)
        x = torch_apply(dct.idct, x)
        return x

    def reset_parameters(self):
        for i in range(self.num_layers):
            init.kaiming_uniform_(self.W[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.B[i], -bound, bound)


lr_rate = 0.001
epochs_num = 20
device = 'cpu' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 100
#train_loader, test_loader = load_mnist_multiprocess(batch_size)
train_loader, test_loader = load_cifar10()

module = new_tNN()
module = module.to(device)

Loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(module.parameters(), lr=lr_rate)

test_loss_epoch = []
test_acc_epoch = []
train_loss_epoch = []
train_acc_epoch = []
time_list = []

# begain train
for epoch in range(epochs_num):
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    module.train()

    pbar_train = pkbar.Pbar(name='Epoch '+str(epoch+1)+' training:', target=50000/batch_size)
    for i, data in enumerate(train_loader):
        img, label = data
        img = cifar_img_processR(img)
        img = raw_img(img, batch_size, n=32)
        img = img.to(device)
        label = label.to(device)

        # forward
        out = module(img)

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
        
        pbar_train.update(i)

    print('[{Epoch}/{Epochs_num}] Loss:{Running_loss} Acc:{Running_acc}'
          .format(Epoch=epoch + 1, Epochs_num=epochs_num, Running_loss=(running_loss / i),
                  Running_acc=running_acc / i))
    train_loss_epoch.append(running_loss / i)
    train_acc_epoch.append((running_acc / i) * 100)

    module.eval()
    eval_loss = 0.0
    eval_acc = 0.0

    pbar_test = pkbar.Pbar(name='Epoch '+str(epoch+1)+' test', target=10000/batch_size)
    for i, data in enumerate(test_loader):
        img, label = data
        img = cifar_img_processR(img)
        img = raw_img(img, batch_size, n=32)
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            out = module(img)
            out = torch.transpose(scalar_tubal_func(out), 0, 1)
            loss = Loss_function(out, label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()

        pbar_test.update(i)

    print('Test Loss: {Eval_loss}, Acc: {Eval_acc}'
          .format(Eval_loss=eval_loss / len(test_loader), 
                  Eval_acc=eval_acc / len(test_loader)))
    test_loss_epoch.append(eval_loss / len(test_loader))
    test_acc_epoch.append((eval_acc / len(test_loader)) * 100)
    time_list.append(time.time() - since)

    if np.isnan(eval_loss):
        print('invalid loss')
        break
