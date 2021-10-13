## MNIST Dataset

Image size: 28 x 28.  

#Epoch: 100; 300 for tNN in [1].

Batch size: 128; 100 for tNN in [1].

Optimizer: Adam.

Rank: 10.

|Networks|File|Layers|Test accuracy|Learning rate|Initialization|
|-|-|-|-|-|-|
|FC-4-layer |fc_4L_mnist.py|[784, 784, 784, 784, 10]|98.64%|0.001|random
|FC-8-layer |fc_8L_mnist.py|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.71%|0.001|random
|FC-4-layer (low-rank)| fc_4L_lowrank_mnist.py|[784, 10, 784, 10, 784, 10, 784, 10]| 96.42%|0.001|xavier normal
|FC-8-layer (low-rank)| fc_8L_lowrank_mnist.py|[784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10]| 96.46%|0.001|xavier normal
|tNN-4-layer |tnn_4L_mnist.py| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.90%;<br> <98.0% in [1].|0.01|random
|tNN-8-layer |tnn_8L_mnist.py| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.59% (28/100 epochs);<br> ~ 98.0% in [1].|0.01; <br>0.1 in [1].|random
|tNN-8-layer-row-7 |tnn_8L_row_7_mnist.py| [(7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 10, 112)]|97.78% |0.001|random
|tNN-8-layer-row-14 |tnn_8L_row_14_mnist.py| [(14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 10, 56)]|97.79% |0.001|random
|tNN-8-layer-row-16 |tnn_8L_row_16_mnist.py| [(16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 10, 49)]|97.86% |0.001|random
|Spectral-tensor-8-layer-subnets-7| spectral_tensor_8L_subnets_7_mnist.py| 7 subnetworks: <br>[112 (28x4), 112, 112, 112, 112, 112, 112, 112, 10] for each subnetwork. | 98.26% |0.001|random
|Spectral-tensor-8-layer-subnets-14| spectral_tensor_8L_subnets_14_mnist.py| 14 subnetworks: <br>[56 (28x2), 56, 56, 56, 56, 56, 56, 56, 10] for each subnetwork. | 98.14% |0.001|random
|Spectral-tensor-8-layer-subnets-16| spectral_tensor_8L_subnets_16_mnist.py| 16 subnetworks: <br>[49 (7x7)), 49, 49, 49, 49, 49, 49, 49, 10] for each subnetwork. | 98.36% |0.001|random
|Spectral-tensor-4-layer-subnets-28| spectral_tensor_4L_subnets_28_mnist.py| 28 subnetworks: <br>[28, 28, 28, 28, 10] for each subnetwork. | 94.53% | 0.001| random
|Spectral-tensor-8-layer-subnets-28| spectral_tensor_8L_subnets_28_mnist.py| 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 94.26% |0.001|random


tNN for row-_x_ images: reorganize each image into a matrix with a row of size _x_ and train the corresponding tNN.


**Our spectral tensor networks with _x_ subnetworks**: 

1). Preprocess training dataset: reorganize each image into a matrix with a row of size _x_, perform DCT on the data along the row-dimension (size _x_), and split the training dataset into _x_ subsets corresponding to _x_ spectrals (for each image, each spectral has a vector); 

2). Train _x_ subnetworks (4-layer and 8-layer FC, respectively) with training dataset: the _x_ spectral data as **input** and the corresponding labels as **output**;

3). Obtain the trained _x_ subnetworks and corresponding loss values; 

4). In the testing phase, use the loss values to set weights as 1/loss; get the _x_ spectrals of a new image and input them into the _x_ trained subnetworks; ensemble the _x_ outputs by weighted sum to obtain the predicted label.


![avatar](./figs/mnist_acc.png)

![avatar](./figs/mnist_loss.png)

![avatar](./figs/spectral_tnn_4L_subnets_28_mnist_acc.png)

![avatar](./figs/spectral_tnn_4L_subnets_28_mnist_loss.png)

![avatar](./figs/spectral_tensor_8L_subnets_7_mnist_acc.png)

![avatar](./figs/spectral_tensor_8L_subnets_7_mnist_loss.png)

- - -

Image size: 28 x 28.  

#Epoch: 300.  

Batch size: 100.

Optimizer: SGD with momentum = 0.9.

|Networks|File|Layers |Test accuracy|Learning rate|Initialization
|-|-|-|-|-|-|
|CNN-4-layer|cnn_4L_mnist.py|[(Conv, ReLU, MaxPool), (Conv, ReLU, Dropout, MaxPool), (Conv, ReLU, MaxPool), (Dropout, Linear)] | 99.44% | 0.01 | random 
|CNN-4-layer|cnn_8L_mnist.py|[(Conv, ReLU), (Conv, ReLU), (Conv, ReLU), (Conv, ReLU, Dropout, MaxPool), (Conv, ReLU), (Conv, ReLU), (Conv, ReLU, MaxPool), (Dropout, Linear)] | 99.47% |  0.01 | random 



## CIFAR 10 Dataset
Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Rank: 16.

Optimizer: SGD with momentum = 0.9.

|Network|File|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|-|
|FC-4-layer|fc_4L_cifar10.py|[3072, 3072, 3072, 3072, 10]|59.40%|0.01|random
|FC-8-layer|fc_8L_cifar10.py|[3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 10]|59.19%|0.01|random
|FC-4-layer (low-rank)|fc_4_lowrank_cifar10.py|[3072, 16, 3072, 16, 3072, 16, 3072, 10]|51.25%(need to be tuned)|0.01|xavier normal
|FC-8-layer (low-rank)|fc_8L_lowrank_cifar10.py|[3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 10]|48.33%(need to be tuned)|0.0001|xavier normal

- - - 

Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Optimizer: SGD with momentum = 0.9 for CNN; Adam for spectral convolutional tensor network.

|Network|File|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|-|
|CNN-4-layer|cnn_4L_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU, MaxPool, Dropout), (Dropout, Linear)] | 87.04% | 0.05 | random
|CNN-8-layer|cnn_8L_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.07% | 0.01 | random
|Spectral-convolutional-tensor-9-layer-subnets-2|spectral_conv_tensor_9L_subnets_2_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 91.37% | 0.001 | random
|Spectral-convolutional-tensor-9-layer-subnets-4|spectral_conv_tensor_9L_subnets_4_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 88.23% | 0.001 | random
|Spectral-convolutional-tensor-9-layer-subnets-8|spectral_conv_tensor_9L_subnets_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 82.03% | 0.001 | random

**Our spectral convolutional tensor networks with _x_ subnetworks**: 

1). Preprocess training dataset: reorganize each image into an image with a row of size _x_, perform DCT on the data along the row-dimension (size _x_), and split the training dataset into _x_ subsets corresponding to _x_ spectrals (for each image, each spectral has a tensor);

2). Train _x_ subnetworks (9-layer CNN) with training dataset: the _x_ spectral data as **input** and the corresponding labels as **output**;

3). Obtain the trained _x_ subnetworks and corresponding loss values;

4). In the testing phase, use the loss values to set weights as 1/loss; get the _x_ spectrals of a new image and input them into the _x_ trained subnetworks; fuse the _x_ outputs by weighted sum to obtain the predicted label.

- - -

Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Optimizer: SGD with momentum = 0.9 for CNN;

|Network|File|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|-|
|Spectral-convolutional-tensor-8-layer-subnets-3| spectral_conv_tensor_8L_subnets_3_cifar10.py|3 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.47% | 0.001 | random 

**Our spectral convolutional tensor networks with 3 subnetworks**: 

1). Preprocess training dataset: perform DCT on the data along the channel-dimension (size 3), and split the training dataset into 3 subsets corresponding to 3 spectrals (for each image, each spectral has a 32 x 32 matrix);

2). Train 3 subnetworks (8-layer CNN) with training dataset: the 3 spectral data as **input** and the corresponding labels as **output**;

3). Obtain the trained 3 subnetworks and corresponding loss values;

4). In the testing phase, use the loss values to set weights as 1/loss; get the 3 spectrals of a new image and input them into the 3 trained subnetworks; fuse the 3 outputs by weighted sum to obtain the predicted label.



![avatar](./figs/fusing_cnn_8L_cifar10_acc.png)

![avatar](./figs/fusing_cnn_8L_cifar10_loss.png)

- - -

Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Optimizer: SGD with momentum = 0.9.

|Network|File|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|-|
|Spectral-convolutional-tensor-8-layer-avg-subnets-4| spectral_conv_tensor_8L_avg_subnets_4_cifar10.py|4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.93% | 0.001 | random 

**Our spectral convolutional tensor networks with 4 subnetworks**:

1). Preprocess training dataset:

* Add a new channel _m_ to each image: [_r_, _g_, _b_] -> [_r_, _g_, _b_, _m_], where _m_ is the average of _r_, _g_, _b_ channels; 

* Reorganize the images by rearranging the 4 channels in 4 ways: [_r_, _g_, _b_, _m_], [_r_, _g_, _m_, _b_], [_r_, _m_, _g_, _b_], [_m_, _r_, _g_, _b_];

* Perform DCT on the 4 rearranged data along the channel-dimension (size 4). For each image the transformed data is a 32 x 32 x 4 tensor with 4 spectrals, where each spectral has a 32 x 32 matrix.

* Split the training dataset into 4 subsets: stack the 4 first spectrals of the 4 transformed data to obtain the first subset; stack the 4 second spectrals of the 4 transformed data to obtain the second subset; ....

2). Train 4 subnetworks (8-layer CNN) with the training datasets: the 4 data subset as **input** and the corresponding labels as **output**;

3). Obtain the trained 4 subnetworks and corresponding loss values;

4). In the testing phase, use the loss values to set weights as 1/loss; get the 4 processed data of a new image (like in step 1)) and input them into the 4 trained subnetworks; fuse the 4 outputs by weighted sum to obtain the predicted label.


![avatar](./figs/spectral_cnn_8L_avg_subnets_4_cifar10_acc.png)

![avatar](./figs/spectral_cnn_8L_avg_subnets_4_cifar10_loss.png)


## Reference

[1] Stable Tensor Neural Networks for Rapid Deep Learning (https://arxiv.org/abs/1811.06569).
