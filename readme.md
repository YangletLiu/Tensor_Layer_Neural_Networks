
## MNIST Task
|Description|File name|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization metod|Optimizer
|-|-|-|-|-|-|-|-|-|
|4-layer FC|fcn_4_mnist.py|[784, 784, 784, 784, 10]|98.63%|100|0.01|64|random|SGD with 0.9 momentum
|8-layer FC|fcn_8_mnist.py|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.66%|100|0.01|64|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|de_fcn_4_mnist.py|[784, 16, 784, 16, 784, 16, 784, 10]| 97.80% |100|0.05|64|xavier normal|SGD with 0.9 momentum
|8-layer FC (decomposed)|de_fcn_8_mnist.py|[784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 10]| 97.86% |100|0.001|64|xavier normal|SGD with 0.9 momentum
|4-layer tNN |tnn_4_mnist.py| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.84%|100|0.1|100|random|SGD with 0.9 momentum
|4-layer tNN **(in reference)**|-| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|<98.0%|300|0.1|100|not mentioned|SGD with 0.9 momentum
|8-layer tNN |tnn_8_mnist.py| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.81%|100|0.1|100|random|SGD with 0.9 momentum
|8-layer tNN **(in reference)**|-| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|~98%|300|0.1|100|not mentioned|SGD with 0.9 momentum

### Experiments on MNIST

![avatar](./figs/mnist_acc.png)

![avatar](./figs/mnist_loss.png)


## CIFAR 10 Task
|Description|File name|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization method|Optimizer
|-|-|-|-|-|-|-|-|-|
|4-layer FC|fcn_4_cifar10.py|[3072, 4096, 2048, 1024, 10]|56.62%|100|0.01|64|random|SGD with 0.9 momentum
|8-layer FC|fcn_8_cifar10.py|[3072, 4096, 4096, 2048, 2048, 1024, 1024, 512, 10]|57.61%|100|0.001|64|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|de_fcn_4_cifar10.py|[3072, 16, 4096, 16, 2048, 16, 1024, 10]|46.75%(need to be tuned)|100|0.005|64|xavier normal|SGD with 0.9 momentum
|8-layer FC (decomposed)|de_fcn_8_cifar10.py|[3072, 16, 4096, 16, 4096, 16, 20416, 16, 20416, 16, 1024, 16, 1024, 16, 512, 10]|44.90%(need to be tuned)|100|0.001|64|xavier normal|SGD with 0.9 momentum
|4-layer CNN|cnn_4_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU, MaxPool, Dropout), (Dropout, Linear)] | 87.04% | 300 | 0.05 | 128 | Random | SGD with 0.9 momentum
|8-layer CNN|cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.70% | 300 | 0.01 | 128 | Random | SGD with 0.9 momentum
|DCT (discrete cosine transform) for input data, build 3 CNNs for the 3 channels of processed data, respectively.|multi_cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] x 3|[channel 1, channel 2, channel 3]  [89.21%, 74.20%, 66.06%]|300|[0.001, 0.001, 0.001]|128|Random|SGD with 0.9 momentum
|1. DCT (discrete cosine transform) for input data, build 3 CNNs for the 3 channels of processed data, respectively. 2. Fuse the outputs of the 3 CNNs (for the 3 channels) for prediction.| fuse_cnn_8_cifar10.py|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] x 3 | 91.47% | 300 | [0.001, 0.001, 0.001] | 128 | Random | SGD with 0.9 momentum


### Experiments on CIFAR 10

![avatar](./figs/fusing_cnn_8_cifar10_acc.png)

![avatar](./figs/fusing_cnn_8_cifar10_loss.png)
