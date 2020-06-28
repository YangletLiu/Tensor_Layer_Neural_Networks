import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

#define params
num_epochs = 100
batch_size = 100
learning_rate = 1e-1

#process img
img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST('./data', transform=img_transform,download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#define the module
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))
        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

loss_epoch = []
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        #forward
        output = model(img)
        loss = criterion(output, img)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_epoch.append(loss)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss))


plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(np.arange(len(loss_epoch)),loss_epoch,label="Loss")
plt.legend()
plt.show()

torch.save(model.state_dict(), './sim_autoencoder.pth')