## MNIST Dataset

Image size: 28 x 28.  

#Epoch: 100; 300 for tNN in [1].

Batch size: 128; 100 for tNN in [1].

Optimizer: Adam.

Rank: 10.

| Networks         | Layers                                                       | Test accuracy | Learning rate | Initialization |
| ---------------- | ------------------------------------------------------------ | ------------- | ------------- | -------------- |
| FC-4L            | [784, 784, 784, 784, 10]                                     | 98.64%        | 0.001         | random         |
| FC-8L            | [784, 784, 784, 784, 784, 784, 784, 784, 10]                 | 98.51%        | 0.001         | random         |
| FC-4L (low-rank) | [784, 10, 784, 10, 784, 10, 784, 10]                         | 96.33%        | 0.001         | random  |
| FC-8L (low-rank) | [784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10] | 97.88%        | 0.001         | random  |
| FC-8L-subnets-28(downsample) | 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 90.60% | 0.001 | random |

## CIFAR 10 Dataset

Image size: 32 x 32 x 3.

Epoch: 300.

Batch size: 128.


| Network     | Layers                                                       | Test accuracy | Learning rate | opt |
| ----------- | ------------------------------------------------------------ | ------------- | ------------- | -------------- |
| CNN-8-layer | [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.38%        | 0.01          | SGD         |
| CNN-8-layer | [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 90.91%        | 0.001          | SGD         |
| CNN-8-layer | [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.42%        | 0.001          | adam        |
| Spectral-CNN-9-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 89.54% (88.79% for 100 epochs)| 0.001 | adam |
| Spectral-CNN-9-layer-subnets-4 | 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 89.42% (88.57 for 100 epochs) | 0.001 | SGD |