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

def build(model_name, num_nets, decomp=True):
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

    if decomp:
        net = decompose_FC(net, mode="low_rank_matrix")
    print("==> Done")
    return net

