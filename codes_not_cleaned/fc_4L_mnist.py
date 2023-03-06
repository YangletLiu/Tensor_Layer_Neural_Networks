######################### 0. import packages #############################
import time
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys


########################## 1. load data ##################################
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 256

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

trainset = datasets.MNIST(root="../datasets", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

testset = datasets.MNIST(root="../datasets", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


########################### 2. define model ##############################
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

########################### 3. build model functions #####################
def build(decomp=True):
    print("==> Building model..")
    full_net = FC4Net(784, 784, 784, 784, 10)
    print("==> Done")
    return full_net


########################### 4. train and test functions ##################
def test(epoch, net, criterion, best_acc, test_acc_list, test_loss_list):
    net.eval()
    test_loss = 0
    correct = 0     # the number of correctly classified images
    total = 0       # the number of images

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        acc = 100.* correct / total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc: %.2f%%   " %(epoch + 1, loss.item(), acc))

        if acc > best_acc:
            best_acc = acc
            # Save checkpoint when best model
            checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_acc": best_acc,
                    }
            torch.save(checkpoint, "../fc_4L_mnist.pth.tar")
        test_acc_list.append(acc)
        test_loss_list.append(test_loss / total)
    return best_acc


def train(num_epochs, net):
    net = net.to(device)
    # initialize some metrices
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = 0.

    start_time = time.time()

    lr0 = 0.01
    current_lr = lr0
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr0)

    try:
        for epoch in range(num_epochs):
            net.train()
            train_loss = 0      # training loss at this epoch
            correct = 0         # the number of correctly classified images
            total = 0           # the number of images

            print("\n=> Training Epoch #%d, LR=%.4f" %(epoch+1, current_lr))
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                optimizer.zero_grad()
                outputs = net(inputs)               # Forward Propagation
                loss = criterion(outputs, targets)  # Loss
                loss.backward()                     # Backward Propagation
                optimizer.step()                    # Optimizer update

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

                sys.stdout.write('\r')
                sys.stdout.write("| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%   "
                        %(epoch+1, num_epochs, batch_idx+1,
                          (len(trainset) // batch_size) + 1, loss.item(), 100. * correct / total))
                sys.stdout.flush()

            best_acc = test(epoch, net, criterion, best_acc, test_acc_list, test_loss_list)
            train_acc_list.append(100. * correct / total)
            train_loss_list.append(train_loss / total)
            now_time = time.time()
            print("| Best Acc: %.2f%% "%(best_acc))
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    print("\nBest training accuracy overall: %.3f%%"%(best_acc))

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


########################### 5. save results ##############################
def save_record_and_draw(train_loss, train_acc, test_loss, test_acc):
    # write csv
    with open("fc_4L_mnist_testloss.csv", "w", newline='', encoding="utf-8") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["Test Loss:"])
        f_csv.writerows(enumerate(test_loss,1))
        f_csv.writerow(["Train Loss:"])
        f_csv.writerows(enumerate(train_loss,1))
        f_csv.writerow(["Test Acc:"])
        f_csv.writerows(enumerate(test_acc,1))
        f_csv.writerow(["Train Acc:"])
        f_csv.writerows(enumerate(train_acc,1))

    # draw picture
    fig = plt.figure(1)
    sub1 = plt.subplot(1, 2, 1)
    plt.sca(sub1)
    plt.title("FC-4L Loss on MNIST ")
    plt.plot(np.arange(len(test_loss)), test_loss, color="red", label="TestLoss",linestyle="-")
    plt.plot(np.arange(len(train_loss)), train_loss, color="blue", label="TrainLoss",linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    sub2 = plt.subplot(1, 2, 2)
    plt.sca(sub2)
    plt.title("FC-4L Accuracy on MNIST ")
    plt.plot(np.arange(len(test_acc)), test_acc, color="green", label="TestAcc",linestyle="-")
    plt.plot(np.arange(len(train_acc)), train_acc, color="orange", label="TrainAcc",linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")

    plt.legend()
    plt.show()

    plt.savefig("./fc_4L_mnist.jpg")


def main():
    net = build(decomp=False)
    print(net)
    train_loss, train_acc, test_loss, test_acc = train(100, net)
    save_record_and_draw(train_loss, train_acc, test_loss, test_acc)


if __name__ == "__main__":
    main()
    