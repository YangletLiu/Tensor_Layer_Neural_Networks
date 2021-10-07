## MNIST Dataset

Image size: 28 x 28.  #Epoch = 100.  Batch size = 64.

Rank = 16.

|Networks|File|Layers|Test accuracy|Learning rate|Initialization|Optimizer
|-|-|-|-|-|-|-|
|FC-4-layer |fc_4_mnist.py|[784, 784, 784, 784, 10]|98.63%|0.01|random|SGD with momentum=0.9
|FC-8-layer |fc_8_mnist.py|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.66%|0.01|random|SGD with momentum=0.9
|FC-4-layer (low-rank)| fc_4_lowrank_mnist.py|[784, 16, 784, 16, 784, 16, 784, 10]| 97.80%|0.05|xavier normal|SGD with momentum=0.9 
|FC-8-layer (low-rank)| fc_8_lowrank_mnist.py|[784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 10]| 97.86%|0.001|xavier normal|SGD with  momentum=0.9

![avatar](./figs/mnist_acc.png)

![avatar](./figs/mnist_loss.png)

Image size: 28 x 28.  #Epoch = 300.  Batch size = 100.

Rank = 16.
|Networks|File|Layers |Test accuracy|Learning rate|Initialization|Optimizer
|-|-|-|-|-|-|-|
|CNN-4-layer|cnn_4_mnist.py|[(Conv, ReLU, MaxPool), (Conv, ReLU, Dropout, MaxPool), (Conv, ReLU, MaxPool), (Dropout, Linear)] | 99.44% | 0.01 | random | SGD with momentum=0.9
|CNN-4-layer|cnn_8_mnist.py|[(Conv, ReLU), (Conv, ReLU), (Conv, ReLU), (Conv, ReLU, Dropout, MaxPool), (Conv, ReLU), (Conv, ReLU), (Conv, ReLU, MaxPool), (Dropout, Linear)] | 99.47% |  0.01 | random | SGD with momentum=0.9
|tNN-4-layer |tnn_4_mnist.py| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.84%|0.1|random|SGD with momentum=0.9
|tNN-4-layer [1]|-| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|<98.0%|0.1|-|SGD with momentum=0.9
|tNN-4-layer |tnn_8_mnist.py| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.81%|0.01|random|SGD with momentum=0.9
|tNN-4-layer [1]|-| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|~98%|0.1|-|SGD with momentum=0.9

[1] Stable Tensor Neural Networks for Rapid Deep Learning (https://arxiv.org/abs/1811.06569)

Image size: 28 x 28.  #Epoch = 300.  Batch size = 100.
|Methods|File|Layers|Test accuracy|Learning rate|Initialization|Optimizer
|-|-|-|-|-|-|-|
|1. DCT (discrete cosine transform) for input data over width-dimension, build 28 4-layer FCNs for the 28 channels of processed data, respectively. 2. Fuse the outputs of the 28 FCNs (for the 28 channels) for prediction.| fuse_spectral_tnn_4_mnist.py|[28, 28, 28, 28, 10] x 28 | 95.74% | 0.001| random | SGD with momentum = 0.9
|1. DCT (discrete cosine transform) for input data over width-dimension, build 28 8-layer FCNs for the 28 channels of processed data, respectively. 2. Fuse the outputs of the 28 FCNs (for the 28 channels) for prediction.| fuse_spectral_tnn_8_mnist.py|[28, 28, 28, 28, 28, 28, 28, 28, 10] x 28 | 95.92% |0.001|random | SGD with momentum = 0.9

![avatar](./figs/spectral_tnn_4_mnist_acc.png)

![avatar](./figs/spectral_tnn_4_mnist_loss.png)



## CIFAR 10 DataSet
Image size: 32 x 32 x 3.  #Epoch = 300.  Batch size = 128.

Rank = 16.

|Network|File|Layers|Test accuracy|Learning rate|Initialization|Optimizer
|-|-|-|-|-|-|-|
|FC-4-layer|fc_4_cifar10.py|[3072, 3072, 3072, 3072, 10]|59.40%|0.01|random|SGD with 0.9 momentum
|FC-8-layer|fc_8_cifar10.py|[3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 10]|59.19%|0.01|random|SGD with momentum = 0.9
|FC-4-layer (decomposed)|fc_4_lowrank_cifar10.py|[3072, 16, 3072, 16, 3072, 16, 3072, 10]|51.25%(need to be tuned)|0.01|xavier normal|SGD with momentum = 0.9
|FC-8-layer (decomposed)|fc_8_lowrank_cifar10.py|[3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 10]|48.33%(need to be tuned)|0.0001|xavier normal|SGD with momentum = 0.9
|CNN-4-layer|cnn_4_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU, MaxPool, Dropout), (Dropout, Linear)] | 87.04% | 0.05 | random | SGD with momentum = 0.9
|CNN-8-layer|cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.07% | 0.01 | random | SGD with momentum = 0.9

Image size: 32 x 32 x 3.  #Epoch = 300.  Batch size = 128.
|Methods|File|Layers|Test accuracy|Learning rate|Initialization|Optimizer
|-|-|-|-|-|-|-|
|DCT (discrete cosine transform) for input data, build 3 CNNs for the 3 channels of processed data, respectively.|multi_cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] x 3|[channel 1, channel 2, channel 3]: [89.21%, 74.20%, 66.06%]|[0.001, 0.001, 0.001]|random|SGD with 0.9 momentum
|1. DCT (discrete cosine transform) for input data, build 3 CNNs for the 3 channels of processed data, respectively. 2. Fuse the outputs of the 3 CNNs (for the 3 channels) for prediction.| fuse_cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] x 3 | 91.47% | [0.001, 0.001, 0.001] | random | SGD with 0.9 momentum


![avatar](./figs/fusing_cnn_8_cifar10_acc.png)

![avatar](./figs/fusing_cnn_8_cifar10_loss.png)

Image size: 32 x 32 x 3.  #Epoch = 300.  Batch size = 128.
|Methods|File|Layers|Test accuracy|Learning rate|Initialization|Optimizer
|-|-|-|-|-|-|-|
|1. Add a new channel _m_ to CIFAR images: [_r_, _g_, _b_] -> [_r_, _g_, _b_, _m_], where _m_ is the average of _r_, _g_, _b_ channels. 2. Reorganize the images by arranging the 4 channels in 4 ways: [_r_, _g_, _b_, _m_], [_r_, _g_, _m_, _b_], [_r_, _m_, _g_, _b_], [_m_, _r_, _g_, _b_]. 3. DCT for the reorganized data after 4 kinds of arrangement, respectively. 4. Stack the 4 first channels of the transformed data to obtain tensor data with size of 32x32x4, and feed the tensor data to one CNN. Stack the 4 second channels ... 5. Fuse the outputs of the 4 CNNs for prediction.| 4_channel_fuse_cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] x 4 | [channel 1, channel 2, channel 3, channel 4]: [89.37%, 78.85%, 79.83%, 79.66%], fusing: 91.93% | [0.001, 0.001, 0.001, 0.001] | random | SGD with momentum = 0.9


![avatar](./figs/spectral_cnn_8_cifar10_acc.png)

![avatar](./figs/spectral_cnn_8_cifar10_loss.png)

