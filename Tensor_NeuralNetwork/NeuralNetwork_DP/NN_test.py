import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

#define super params
batch_size =64
learning_rate =1e-2
num_epochs =100

#download MNIST
train_datset =datasets.FashionMNIST(
    root='../datasets',train=True,transform=transforms.ToTensor(),download=True)

test_dataset = datasets.FashionMNIST(
    root='../datasets',train=False,transform=transforms.ToTensor(),download=False)

#define dataset loader
train_loader =DataLoader(train_datset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#define NN model
class neuralNetwork(nn.Module):
    """in_dem represents the dimension of input
        n_hidden_1,n_hidden_2,n_hidden_3 denotes the three hidden layers' number"""
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,n_hidden_3,out_dim):
        super(neuralNetwork, self).__init__()
        #layer1
        self.layer1 = nn.Sequential(
            #linear A*W+b
            nn.Linear(in_dim,n_hidden_1),
            #activate function ReLU
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1,n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3,out_dim),
            nn.ReLU(True)
        )

    #forward
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

#define NN model
model = neuralNetwork(28*28,300,200,100,10)

#test if GPU is available
use_gpu =torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

#define loss and optimizer
#CrossEntropyLoss Function
criterion = nn.CrossEntropyLoss()
#SGD
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

loss_after_epoch = []
acc_after_epoch = []

#begain train
for epoch in range(num_epochs):
    print('*'*10)
    print(f'epoch {epoch+1}')
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0

    model.train()
    for i,data in enumerate(train_loader,1):
        #load every single train sample img
        img,label =data
        img = img.view(img.size(0),-1)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        #forward
        out = model(img)
        loss = criterion(out,label)
        running_loss += loss.item()
        _,pred = torch.max(out,1)
        running_acc += (pred == label).float().mean()

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%500 == 0:
            print(f'[{epoch+1}/{num_epochs}] Loss:{running_loss/i:.6f} Acc:{running_acc/i:.6f}')
    print(f'Finish {epoch+1} epoch, Loss:{running_loss/i:.6f}, Acc:{running_acc/i:.6f}')

    #model evaluate
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0

    for data in test_loader:
        img,label = data
        img = img.view(img.size(0),-1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        with torch.no_grad():
            out = model(img)
            loss = criterion(out,label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()
    print(f'Test Loss: {eval_loss / len(test_loader):.6f}, Acc: {eval_acc / len(test_loader):.6f}')
    print(f'Time:{(time.time() - since):.1f} s')
    loss_after_epoch.append(eval_loss)
    acc_after_epoch.append(eval_acc)

#save the model
torch.save(model.state_dict(),'./NeuralNetwork.pth')

fig = plt.figure(figsize=(20,10))
plt.plot(np.arange(len(loss_after_epoch)),loss_after_epoch,'+',label='Loss')
plt.plot(np.arange(len(acc_after_epoch)),acc_after_epoch,'d',label='Acc')
plt.legend()
plt.show()




