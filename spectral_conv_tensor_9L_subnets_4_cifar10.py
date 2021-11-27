######################### 0. import packages #############################
import time
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
import torch_dct


########################## 1. load data ####################################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010]),
                                ])

transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
                                ])

num_nets = 4
batch_size = 128
trainset = datasets.CIFAR10(root='../datasets', train=True, transform=transform_train, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
num_train = len(trainset)

testset = datasets.CIFAR10(root='../datasets', train=False, transform=transform_test, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
num_test = len(testset)


########################### 2. define model ##################################
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
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
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
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.pred = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8192*2, 10),
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
        x = self.conv9(x)
        x = x.view(x.size(0),-1)
        x = self.pred(x)
        return x


######################## 3. build model functions #################
def downsample_img(img, block_size):
    batch_, c_, m_, n_ = img.shape
    row_step, col_step = block_size
    row_blocks = m_ // row_step
    col_blocks = n_ // col_step
    assert num_nets == row_step * col_step, "the number of downsampled images is not equal to the number of num_nets"
    assert m_ % row_step == 0, "the image can' t be divided into several downsample blocks in row-dimension"
    assert n_ % col_step == 0, "the image can' t be divided into several downsample blocks in col-dimension"

    # show_mnist_fig(img[0, 0, :, :], "split_image_seg{}.png".format(num_nets))

    components = []
    for row in range(row_step):
        for col in range(col_step):
            components.append(img[:, :, row::row_step, col::col_step].unsqueeze(dim=-1))
    img = torch.cat(components, dim=-1)

    # for i in range(row_step * col_step):
    #     show_mnist_fig(img[0, 0, :, :, i], "split_image_seg{}.png".format(i))

    return img


def preprocess_cifar10(img, block_size):
    img = downsample_img(img, block_size=block_size)
    img = torch_dct.dct(img)
    return img



# build model
def build(decomp=False):
    print('==> Building model..')
    full_net = CNN8CIFAR10()
    if decomp:
        raise("No Tensor Neural Network decompostion implementation.")
    print('==> Done')
    return full_net


########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss().to(device)
lr0 = [0.001] * num_nets
fusing_plan = [list(range(num_nets))]
fusing_num = len(fusing_plan)


def query_lr(epoch):
    lr = lr0
    return lr


def set_lr(optimizer, epoch):
    current_lr = query_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def set_weight(train_loss, mode="one_over"):
    fusing_weight = [1.] * len(train_loss)
    weight_sum = 0.
    rank_list = np.argsort(train_loss)
    p = 0.3

    for idx in range(len(train_loss)):
        if mode == "one_over":
            fusing_weight[idx] = 1/train_loss[idx]
            weight_sum += fusing_weight[idx]
        elif mode == "one_over_square":
            fusing_weight[idx] = 1/train_loss[idx]
            weight_sum += fusing_weight[idx]
        elif mode == "geometry":
            fusing_weight[rank_list[idx]] = p * np.power((1 - p), idx)
            weight_sum += fusing_weight[rank_list[idx]]

    fusing_weight = torch.tensor(fusing_weight, device=device)
    fusing_weight /= weight_sum
    return fusing_weight


def test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list, fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_weight=None):
    for net in nets:
        net.eval()
    test_loss = [0] * num_nets
    correct = [0] * num_nets
    total = [0] * num_nets
    outputs = [0] * num_nets

    fusing_test_loss = [0] * fusing_num
    fusing_correct = [0] * fusing_num
    fusing_total = [0] * fusing_num
    fusing_outputs = [0] * fusing_num
    if fusing_weight == None:
        fusing_weight = [1. / num_nets] * num_nets

    with torch.no_grad():
        for batch_idx, (img, targets) in enumerate(testloader):
            img, targets = img.to(device), targets.to(device)
            img = preprocess_cifar10(img, block_size=(2, 2))

            for i in range(num_nets):
                outputs[i] = nets[i](img[:, :, :, :, i])
                loss = criterion(outputs[i], targets)

                test_loss[i] += loss.item()
                _, predicted = torch.max(outputs[i].data, 1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets.data).cpu().sum().item()

            #########################################
            ########### Fusing 2 networks ###########
            for plan_id in range(fusing_num):
                fusing_outputs[plan_id] = 0.
                fusing_weight_sum = 0.
                for net_idx in fusing_plan[plan_id]:
                    fusing_outputs[plan_id] += fusing_weight[net_idx] * outputs[net_idx]
                    fusing_weight_sum += fusing_weight[net_idx]
                fusing_outputs[plan_id] /= fusing_weight_sum

                fusing_loss = criterion(fusing_outputs[plan_id], targets)

                fusing_test_loss[plan_id] += fusing_loss.item()
                _, predicted = torch.max(fusing_outputs[plan_id].data, 1)
                fusing_total[plan_id] += targets.size(0)
                fusing_correct[plan_id] += predicted.eq(targets.data).cpu().sum().item()

            #########################################
            #########################################
        # Save checkpoint when best model
        acc = [0] * num_nets
        for i in range(num_nets):
            test_loss[i] /= num_test
            acc[i] = 100. * correct[i] / total[i]

        fusing_acc = [0] * fusing_num
        for i in range(fusing_num):
            fusing_test_loss[i] /= num_test
            fusing_acc[i] = 100. * fusing_correct[i] / fusing_total[i]

        print("\n| Validation Epoch #%d\t\tLoss: [%.4f, %.4f, %.4f, %.4f] Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%]   " 
              %(epoch+1, test_loss[0], test_loss[1], test_loss[2], test_loss[3], acc[0], acc[1], acc[2], acc[3]))

        print("| Fusing Loss: [%.4f] Fusing Acc: [%.2f%%]   "%(fusing_test_loss[0], fusing_acc[0]))

        for i in range(num_nets):
            if acc[i] > best_acc[i]:
                best_acc[i] = acc[i]
                # torch.save(nets[i], "./multi_cnn_8_channel_{}.pth".format(i))

        for i in range(fusing_num):
            if fusing_acc[i] > best_fusing_acc[i]:
                best_fusing_acc[i] = fusing_acc[i]
        
        test_acc_list.append(acc)
        test_loss_list.append([test_loss[i] for i in range(num_nets)])

        fusing_test_acc_list.append(fusing_acc)
        fusing_test_loss_list.append([fusing_test_loss[i] for i in range(fusing_num)])
    return best_acc, best_fusing_acc


# Training
def train_multi_nets(num_epochs, nets):
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = [0.] * num_nets

    fusing_train_acc_list, fusing_train_loss_list = [], []
    fusing_test_acc_list, fusing_test_loss_list = [], []
    best_fusing_acc = [0.] * fusing_num

    fusing_weight = set_weight([1]*num_nets, mode="one_over")

    start_time = time.time()

    optimizers = []
    for i in range(num_nets):
        nets[i] = nets[i].to(device)
        # optimizers.append(torch.optim.SGD(nets[i].parameters(), lr=lr0[i], momentum=0.9))
        optimizers.append(torch.optim.Adam(nets[i].parameters(), lr=lr0[i]))

    current_lr = lr0

    try:
        for epoch in range(num_epochs):
            for net in nets:
                net.train()
            train_loss = [0] * num_nets
            correct = [0] * num_nets
            total = [0] * num_nets
            loss = [0] * num_nets
            fusing_train_loss = 0.
            fusing_train_correct = 0
            fusing_train_total = 0

            print('\n=> Training Epoch #%d, LR=[%.4f, %.4f, %.4f, %.4f]' %(epoch+1, current_lr[0], current_lr[1], current_lr[2], current_lr[3]))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                inputs = preprocess_cifar10(inputs, block_size=(2, 2))

                fusing_output = 0.
                for net_idx in range(num_nets):
                    optimizers[net_idx].zero_grad()
                    outputs = nets[net_idx](inputs[:, :, :, :, net_idx])               # Forward Propagation
                    loss[net_idx] = criterion(outputs, targets)  # Loss
                    loss[net_idx].backward()  # Backward Propagation
                    optimizers[net_idx].step()  # Optimizer update

                    train_loss[net_idx] += loss[net_idx].item()
                    _, predicted = torch.max(outputs.data, 1)
                    total[net_idx] += targets.size(0)
                    correct[net_idx] += predicted.eq(targets.data).cpu().sum().item()

                    fusing_output += fusing_weight[net_idx] * outputs.detach()

                fusing_train_loss += criterion(fusing_output, targets).item()
                _, predicted = torch.max(fusing_output.data, 1)
                fusing_train_total += targets.size(0)
                fusing_train_correct += predicted.eq(targets.data).cpu().sum().item()
                fusing_weight = set_weight(train_loss, mode="geometry")
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: [%.4f, %.4f, %.4f, %.4f] Acc: %.3f%%, [%.3f%%, %.3f%%, %.3f%%, %.3f%%]   '
                        %(epoch+1, num_epochs, batch_idx+1,
                          math.ceil(len(trainset)/batch_size), loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(),
                          100.*fusing_train_correct/fusing_train_total, 100.*correct[0]/total[0], 100.*correct[1]/total[1], 100.*correct[2]/total[2], 100.*correct[3]/total[3]))
                sys.stdout.flush()

            fusing_weight = set_weight(train_loss, "geometry")
            best_acc, best_fusing_acc = test_fusing_nets(epoch, nets, best_acc, best_fusing_acc, test_acc_list,
                                        fusing_test_acc_list, test_loss_list, fusing_test_loss_list, fusing_weight=fusing_weight)

            train_acc_list.append([100.*correct[i]/total[i] for i in range(num_nets)])
            train_loss_list.append([train_loss[i] / num_train for i in range(num_nets)])
            fusing_train_loss_list.append(fusing_train_loss)
            fusing_train_acc_list.append(100.*fusing_train_correct/fusing_train_total)
            now_time = time.time()
            print("| Best Acc: [%.2f%%, %.2f%%, %.2f%%, %.2f%%] "%(best_acc[0], best_acc[1], best_acc[2], best_acc[3]))
            print("| Best Fusing Acc: [%.2f%%] "%(best_fusing_acc[0]))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    print("\nBest training accuracy overall: [%.3f%%, %.3f%%, %.3f%%, %.3f%%] "%(best_acc[0], best_acc[1], best_acc[2], best_acc[3]))
    print("| Best fusing accuracy overall: [%.2f%%] "%(best_fusing_acc[0]))

    return train_loss_list, train_acc_list, fusing_train_loss_list, fusing_train_acc_list, test_loss_list, test_acc_list, fusing_test_loss_list, fusing_test_acc_list


def save_record_and_draw(train_loss, train_acc, fusing_train_loss, fusing_train_acc, test_loss, test_acc, fusing_test_loss, fusing_test_acc):

    # write csv
    with open('spectral_conv_tensor_9L_subnets_4_cifar10_testloss.csv','w',newline='',encoding='utf-8') as f:
        f_csv = csv.writer(f)

        f_csv.writerow(["Test Acc:"])
        for idx in range(len(test_acc)):
            f_csv.writerow([idx + 1] + test_acc[idx])

        f_csv.writerow(["Test Loss:"])
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + test_loss[idx])

        f_csv.writerow(["Fusing Test Acc:"] + fusing_plan)
        for idx in range(len(fusing_test_acc)):
            f_csv.writerow([idx + 1] + fusing_test_acc[idx])

        f_csv.writerow(["Fusing Test Loss:"] + fusing_plan)
        for idx in range(len(test_loss)):
            f_csv.writerow([idx + 1] + fusing_test_loss[idx])
            
        f_csv.writerow(["Train Acc"])
        for idx in range(len(train_acc)):
            f_csv.writerow([idx + 1] + train_acc[idx])

        f_csv.writerow(["Train Loss"])
        for idx in range(len(train_loss)):
            f_csv.writerow([idx + 1] + train_loss[idx])

        f_csv.writerow(["Fusing Train Acc:"])
        for idx in range(len(fusing_train_acc)):
            f_csv.writerow([idx + 1] + [fusing_train_acc[idx]])

        f_csv.writerow(["Fusing Train Loss:"])
        for idx in range(len(fusing_train_loss)):
            f_csv.writerow([idx + 1] + [fusing_train_loss[idx]])

    # draw picture
    test_acc = np.array(test_acc)
    test_loss = np.array(test_loss)
    fusing_test_acc = np.array(fusing_test_acc)
    fusing_test_loss = np.array(fusing_test_loss)
    train_acc = np.array(train_acc)
    train_loss = np.array(train_loss)
    fusing_train_acc = np.array(fusing_train_acc)
    fusing_train_loss = np.array(fusing_train_loss)

    plt.cla()
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title('spectral-conv-tensor-9L-subnets-4 Loss on CIFAR10 ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_loss[:, i])), test_loss[:, i], label='TestLoss_{}'.format(i+1),linestyle='-')
    for i in range(fusing_num):
        plt.plot(np.arange(len(fusing_test_loss[:, i])), fusing_test_loss[:, i], label='FusingTestLoss_{}'.format(i+1),linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_loss[:, i])), train_loss[:, i], label='TrainLoss_{}'.format(i+1),linestyle='--')
    plt.plot(np.arange(len(fusing_train_loss[:])), fusing_train_loss[:], label='FusingTrainLoss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title('spectral-conv-tensor-9L-subnets-4 Accuracy on CIFAR10 ')
    for i in range(num_nets):
        plt.plot(np.arange(len(test_acc[:, i])), test_acc[:, i], label='TestAcc_{}'.format(i+1),linestyle='-')
    for i in range(fusing_num):
        plt.plot(np.arange(len(fusing_test_acc[:, i])), fusing_test_acc[:, i], label='FusingTestAcc_{}'.format(i+1),linestyle='-')
    for i in range(num_nets):
        plt.plot(np.arange(len(train_acc[:, i])), train_acc[:, i], label='TrainAcc_{}'.format(i+1),linestyle='--')
    plt.plot(np.arange(len(fusing_train_acc[:])), fusing_train_acc[:], label='FusingTrainAcc',linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.legend()
    plt.show()

    plt.savefig('./spectral_conv_tensor_9L_subnets_4_cifar10_trainloss.jpg')


if __name__ == "__main__":
    raw_nets = []
    for _ in range(num_nets):
        raw_nets.append(build(decomp=False))
    print(raw_nets[0])
    train_loss_, train_acc_, fusing_train_loss_, fusing_train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_ = train_multi_nets(300, raw_nets)
    save_record_and_draw(train_loss_, train_acc_, fusing_train_loss_, fusing_train_acc_, test_loss_, test_acc_, fusing_test_loss_, fusing_test_acc_)
