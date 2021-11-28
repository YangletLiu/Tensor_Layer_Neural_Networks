######################### 0. import packages #############################
import time
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import csv
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


# define nn module
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

# dct at the beginning and idct at the end

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

    if torch.__version__ > "1.7.1":
        Vc = torch.view_as_real(torch.fft.fft(v))
    else:
        Vc = torch.rfft(v, 1, onesided=False)

    k = (- torch.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)).to(device)
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

    k = (torch.arange(x_shape[-1], dtype=X.dtype)[None, :] * np.pi / (2 * N)).to(device)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    if torch.__version__ > "1.7.1":
        v = torch.fft.ifft(torch.view_as_complex(V)).real
    else:
        v = torch.irfft(V, 1, onesided=False)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


# circulant convolution part
def torch_tensor_Bcirc(tensor, l, m, n):
    bcirc_A = []
    for i in range(l):
        bcirc_A.append(torch.roll(tensor, shifts=i, dims=0))
    return torch.cat(bcirc_A, dim=2).reshape(l * m, l * n)


def dct_tensor_product(tensorA, tensorB):
    a_l, a_m, a_n = tensorA.shape
    b_l, b_m, b_n = tensorB.shape
    dct_a = torch.transpose(dct(torch.transpose(tensorA, 0, 2)), 0, 2)
    # print(dct_a)
    dct_b = torch.transpose(dct(torch.transpose(tensorB, 0, 2)), 0, 2)
    # print(dct_b)

    dct_product = []
    for i in range(a_l):
        dct_product.append(torch.mm(dct_a[i, :, :], dct_b[i, :, :]))
    dct_products = torch.stack(dct_product)

    idct_product = torch.transpose(idct(torch.transpose(dct_products, 0, 2)), 0, 2).reshape(a_l, a_m, b_n)

    return idct_product


# Loss function(scalar tubal softmax function)
def h_func_dct(lateral_slice):
    l, m, n = lateral_slice.shape

    dct_slice = dct(lateral_slice)

    tubes = [dct_slice[i, :, 0] for i in range(l)]

    # todo: parallelism here, use tensor's batch operation
    h_tubes = []
    for tube in tubes:
        tube_sum = torch.sum(torch.exp(tube))
        h_tubes.append(torch.exp(tube) / tube_sum)
    #######################################################

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
def raw_img(img, seg_length=0):
    """
        :param img: (batch_size, channel=1, n1, n2)
        :return (n2, n1, batch_size)
    """
    img = torch.squeeze(img)
    if seg_length:
        sp = img.shape
        img = img.reshape(sp[0], -1, seg_length)
    ultra_img = img.permute([2, 1, 0])
    return ultra_img


# build model
def build(decomp=False):
    print('==> Building model..')
    full_net = tNN4MNIST()
    if decomp:
        raise("No Tensor Neural Network decompostion implementation.")
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
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (img, targets) in enumerate(testloader):
            img = raw_img(img)
            img, targets = img.to(device), targets.to(device)

            outputs = net(img) / 1e8
            outputs = torch.transpose(scalar_tubal_func(outputs), 0, 1)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

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

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr0)
    current_lr = lr0

    try:
        for epoch in range(num_epochs):
            net.train()
            train_loss = 0
            correct = 0
            total = 0

            # current_lr = set_lr(optimizer, epoch)  # comment this code if use fixed learning rate
            print('\n=> Training Epoch #%d, LR=%.4f' %(epoch+1, current_lr))
            for batch_idx, (img, targets) in enumerate(trainloader):
                img, targets = img.to(device), targets.to(device)
                img = raw_img(img)
                optimizer.zero_grad()

                outputs = net(img) / 1e8
                # softmax function
                outputs = torch.transpose(scalar_tubal_func(outputs), 0, 1)

                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    print("Train loss is nan! Skip this iteration.")
                    continue
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%   '
                        %(epoch+1, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
                sys.stdout.flush()

            best_acc = test(epoch, net, best_acc, test_acc_list, test_loss_list)
            if np.isnan(test_loss_list[-1]):
                print("Test Loss is nan!")
                break
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
    with open('tnn_4L_mnist_testloss.csv','w',newline='',encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(enumerate(test_acc,1))
        f_csv.writerow(['Test Loss:'])
        f_csv.writerows(enumerate(test_loss,1))
        f_csv.writerow(['Train Acc:'])
        f_csv.writerows(enumerate(train_acc,1))
        f_csv.writerow(['Train Loss:'])
        f_csv.writerows(enumerate(train_loss,1))

    # draw picture
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title('tNN-4 Loss on MNIST ')
    plt.plot(np.arange(len(test_loss)), test_loss, color='red', label='TestLoss',linestyle='-')
    plt.plot(np.arange(len(train_loss)), train_loss, color='blue', label='TrainLoss',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('tNN-4 Accuracy on MNIST ')
    plt.plot(np.arange(len(test_acc)), test_acc, color='green', label='TestAcc',linestyle='-')
    plt.plot(np.arange(len(train_acc)), train_acc, color='orange', label='TrainAcc',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./tnn_4L_mnist.jpg')


if __name__ == "__main__":
    raw_net = build(decomp=False)
    print(raw_net)
    train_loss_, train_acc_, test_loss_, test_acc_ = train(100, raw_net)
    save_record_and_draw(train_loss_, train_acc_, test_loss_, test_acc_)
