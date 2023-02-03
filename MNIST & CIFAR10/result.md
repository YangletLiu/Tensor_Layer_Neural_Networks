## MNIST Dataset

Image size: 28 x 28.  

#Epoch: 100; 300 for tNN in [1].

Batch size: 128; 100 for tNN in [1].

Optimizer: Adam.

Rank: 10.

| Networks         | Layers                                                       | Test accuracy | Learning rate | remark |
| ---------------- | ------------------------------------------------------------ | ------------- | ------------- | -------------- |
| FC-4L            | [784, 784, 784, 784, 10]                                     | 98.64%        | 0.001         | -         |
| FC-8L            | [784, 784, 784, 784, 784, 784, 784, 784, 10]                 | 98.51%        | 0.001         | -         |
| FC-4L (low-rank) | [784, 10, 784, 10, 784, 10, 784, 10]                         | 96.33%        | 0.001         | -  |
| FC-8L (low-rank) | [784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10] | 97.88%        | 0.001         | -  |
| FC-8L-subnets-28(downsample) | 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 90.60% | 0.001 | - |
| Spectral-FC-8L-subnets-28 | 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 94.70% | 0.001 | random |
| Spectral-FC-8L-subnets-4(downsample)* | 4 subnetworks: <br>[196, 196, 196, 196, 196, 196, 196, 196, 10] for each subnetwork. | 98.39%  | 0.001 | 98.53% for sub0 |
| Spectral-FC-8L-subnets-4(downsample) | 4 subnetworks: <br>[196, 196, 196, 196, 196, 196, 196, 196, 10] for each subnetwork. | 98.37%  | 0.001 | adamw with 0.01 wd |
| Spectral-FC-8L-subnets-16(downsample) | 16 subnetworks: <br>[49, 49, 49, 49, 49, 49, 49, 49, 10] for each subnetwork. | 97.84% | 0.001 | - |
| Spectral-FC-8L-subnets-28(downsample) | 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 96.93% | 0.001 | - |
## CIFAR 10 Dataset

Image size: 32 x 32 x 3.

Epoch: 300.

Batch size: 128.


| Network     | Layers                                                       | Test accuracy | Learning rate | opt |
| ----------- | ------------------------------------------------------------ | ------------- | ------------- | -------------- |
| CNN-8-layer | [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.38%        | 0.01          | SGD         |
| CNN-8-layer | [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 90.91%        | 0.001          | SGD         |
| CNN-8-layer | [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.42% (90.96% for 100 epochs)  | 0.001          | adam        |
| Spectral-CNN-9-layer-subnets-2 | 2 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.38% (90.67% for 100 epochs)| 0.001 | adam |
| Spectral-CNN-9-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 89.54% (88.79% for 100 epochs)| 0.001 | adam |
| Spectral-CNN-9-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.86% (88.79% for 100 epochs)| 0.001 | adam with steplr scheduler |
| Spectral-CNN-9-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 89.42% (88.57% for 100 epochs) | 0.001 | SGD |
| Spectral-CNN-9-layer-subnets-16 | 16 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 80.27% (78.59% for 100 epochs) | 0.001 | adam |
| Spectral-CNN-8-layer-subnets-2 | 2 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 90.61 % (88.04% for 100 epochs) | 0.001 | adam |
| Spectral-CNN-8-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 89.39 % (88.89% for 100 epochs) | 0.001 | adam |
| Spectral-CNN-8-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 89.54 % (88.93% for 100 epochs) | 0.001 | adamw with 0.01 wd |
| Spectral-CNN-8-layer-subnets-16 | 16 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 80.18%  (79.89% for 100 epochs) | 0.001 | adam |