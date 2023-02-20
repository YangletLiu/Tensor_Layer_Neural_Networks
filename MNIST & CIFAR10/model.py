import torch
from torch import nn
from utils import decompose_FC, dct_tensor_product
import torch.nn.functional as F


class FC4Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(FC4Net, self).__init__()
        # layer1
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, out_dim),
        )

    # forward
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class FC8Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3,
                 n_hidden_4, n_hidden_5, n_hidden_6, n_hidden_7, out_dim):
        super(FC8Net, self).__init__()
        # layer1
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_5),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(n_hidden_5, n_hidden_6),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(n_hidden_6, n_hidden_7),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1)
        )
        self.layer8 = nn.Sequential(
            nn.Linear(n_hidden_7, out_dim),
        )

    # forward
    def forward(self, x):
        x = x.view(x.size(0), -1)

        x1 = self.layer1(x)
        # x = torch.cat((self.layer2(x), x1), dim=-1)

        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)
        # x3  = self.layer5(x)
        x = self.layer5(x) + x3
        x5 = x
        x = self.layer6(x)
        # x = self.layer7(x)
        x = self.layer7(x) + x5
        x = self.layer8(x)
        return x

class tNN4MNIST(nn.Module):
    def __init__(self):
        super(tNN4MNIST, self).__init__()
        """
        use the nn.Parameter() and 'requires_grad = True' 
        to customize parameters which are needed to optimize
        """
        std = 1
        self.W_1 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_1 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_2 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_2 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_3 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_3 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_4 = nn.Parameter(torch.randn(28, 10, 28) * std)
        self.B_4 = nn.Parameter(torch.randn(28, 10, 1) * std)

    def forward(self, x):
        """
        torch_tensor_product is redefined by torch to complete the tensor-product process
        :param x: x is the input 3D-tensor with shape(l,m,n)
                     'n' denotes the batch_size
        :return: this demo defines an one-layer networks,
                    whose output is processed by one-time tensor-product and activation
        """
        x = dct_tensor_product(self.W_1, x) + self.B_1
        x = F.relu(x)
        x = dct_tensor_product(self.W_2, x) + self.B_2
        x = F.relu(x)
        x = dct_tensor_product(self.W_3, x) + self.B_3
        x = F.relu(x)
        x = dct_tensor_product(self.W_4, x) + self.B_4
        return x

class tNN8MNIST(nn.Module):
    def __init__(self):
        super(tNN8MNIST, self).__init__()
        """
        use the nn.Parameter() and 'requires_grad = True' 
        to customize parameters which are needed to optimize
        """
        std = 1
        self.W_1 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_1 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_2 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_2 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_3 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_3 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_4 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_4 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_5 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_5 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_6 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_6 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_7 = nn.Parameter(torch.randn(28, 28, 28) * std)
        self.B_7 = nn.Parameter(torch.randn(28, 28, 1) * std)
        self.W_8 = nn.Parameter(torch.randn(28, 10, 28) * std)
        self.B_8 = nn.Parameter(torch.randn(28, 10, 1) * std)

    def forward(self, x):
        """
        torch_tensor_product is redefined by torch to complete the tensor-product process
        :param x: x is the input 3D-tensor with shape(l,m,n)
                     'n' denotes the batch_size
        :return: this demo defines an one-layer networks,
                    whose output is processed by one-time tensor-product and activation
        """
        x = dct_tensor_product(self.W_1, x) + self.B_1
        x = F.relu(x)
        x = dct_tensor_product(self.W_2, x) + self.B_2
        x = F.relu(x)
        x = dct_tensor_product(self.W_3, x) + self.B_3
        x = F.relu(x)
        x = dct_tensor_product(self.W_4, x) + self.B_4
        x = F.relu(x)
        x = dct_tensor_product(self.W_5, x) + self.B_5
        x = F.relu(x)
        x = dct_tensor_product(self.W_6, x) + self.B_6
        x = F.relu(x)
        x = dct_tensor_product(self.W_7, x) + self.B_7
        x = F.relu(x)
        x = dct_tensor_product(self.W_8, x) + self.B_8
        return x

class CNN8MNIST(nn.Module):
    def __init__(self):
        super(CNN8MNIST,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
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
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.pred = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16*28*28,10)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x

class CNN8CIFAR10(nn.Module):
    def __init__(self):
        super(CNN8CIFAR10,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
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
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x

class CNN10ImageNet(nn.Module):
    def __init__(self):
        super(CNN10ImageNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
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
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
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
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
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
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=0.05),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.pred = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4608, 1000),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x

class CNN9CIFAR10(nn.Module):
    def __init__(self):
        super(CNN8CIFAR10,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
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
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
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
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
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
            # nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.pred = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 10),
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x

def build(model_name, num_nets=0, decomp=True):
    print("==> Building model..")
    weight = 784 // num_nets if num_nets > 0 else 784
    print(weight)
    if model_name == "FC4Net":
        net = FC4Net(weight, weight, weight, weight, 10)

    elif model_name == "FC8Net":
        net = FC8Net(weight, weight, weight, weight, weight, weight, weight, weight, 10)

    elif model_name == "tNN4MNIST":
        net = tNN4MNIST()

    elif model_name == "tNN8MNIST":
        net = tNN8MNIST()

    elif model_name == "CNN8MNIST":
        net = CNN8MNIST()

    elif model_name == "CNN8CIFAR10":
        net = CNN8CIFAR10()

    elif model_name == "CNN9CIFAR10":
        net = CNN9CIFAR10()

    elif model_name == "DenseNetFC":
        net = DenseNetFCN()

    if decomp:
        net = decompose_FC(net, mode="low_rank_matrix")
    print("==> Done")
    return net



