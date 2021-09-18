
## MNIST Task
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization metod|Optimizer
|-|-|-|-|-|-|-|-|
|4-layer FC|[784, 784, 784, 784, 10]|98.63%|100|0.01|64|random|SGD with 0.9 momentum
|8-layer FC|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.66%|100|0.01|64|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|[784, 16, 784, 16, 784, 16, 784, 10]| 97.80% |100|0.05|64|low rank matrix decomposition initialization|SGD with 0.9 momentum
|8-layer FC (decomposed)|[784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 16, 784, 10]| 97.86% |100|0.001|64|low rank matrix decomposition initialization|SGD with 0.9 momentum
|4-layer tNN **(still running)**| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.9%|7/100|0.01|100|random|SGD with 0.9 momentum
|8-layer tNN **(still running)**| [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|86.80%|3/100|0.01|100|random|SGD with 0.9 momentum

## CIFAR 10 Task
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization method|Optimizer
|-|-|-|-|-|-|-|-|
|4-layer FC|[3072, 4096, 2048, 1024, 10]|56.62%|100|0.01|64|random|SGD with 0.9 momentum
|8-layer FC|[3072, 4096, 4096, 2048, 2048, 1024, 1024, 512, 10]|57.61%|100|0.001|64|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|[3072, 16, 4096, 16, 2048, 16, 1024, 10]|46.75%(need to be tuned)|100|0.005|64|low rank matrix decompostion initialization|SGD with 0.9 momentum
|8-layer FC (decomposed)|[3072, 16, 4096, 16, 4096, 16, 20416, 16, 20416, 16, 1024, 16, 1024, 16, 512, 10]|44.90%(need to be tuned)|100|0.001|64|low rank matrix decomposition initialization|SGD with 0.9 momentum


![avatar](./figs/mnist_acc.png)
<!-- ![avatar](./figs/mnist_loss.png) -->
##  File structure
> fcn_4_mnist.py <br>
> fcn_8_mnist.py <br>
> de_fcn_4_mnist.py <br>
> de_fcn_8_mnist.py <br>
> fcn_4_cifar10.py <br>
> fcn_8_cifar10.py <br>
> de_fcn_4_cifar10.py <br>
> de_fcn_8_cifar10.py <br>

