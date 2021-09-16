
## MNIST Task
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization metod|Optimizer
|-|-|-|-|-|-|-|-|
|4-layer FC|[784, 784, 784, 784, 10]|98.40%|100|0.01|64|random|SGD with 0.9 momentum
|8-layer FC|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.56%|100|0.01|64|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|[784, 8, 784, 8, 784, 8, 784, 8, 10]|95.62%|100|0.01|64|low rank matrix decomposition initialization|SGD with 0.9 momentum
|8-layer FC (decomposed)|[784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 10]| **not test yet**|100|0.01|64|low rank matrix decomposition initialization|SGD with 0.9 momentum

## CIFAR 10 Task
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization method|Optimizer
|-|-|-|-|-|-|-|-|
|4-layer FC|[3072, 4096, 2048, 1024, 10]|56.32%|100|1e-4|128|random|Adam
|8-layer FC|[3072, 4096, 4096, 2048, 2048, 1024, 1024, 512, 10]|56.61%|100|1e-3|128|random|Adam
|4-layer FC (decomposed)|[3072, 8, 4096, 8, 2048, 8, 1024, 8, 10]|**not test yet**|
|8-layer FC (decomposed)|[3072, 8, 4096, 8, 4096, 8, 2048, 8, 2048, 8, 1024, 8, 1024, 8, 512, 8, 10]|**not test yet**|

##  File structure
> fcn_4_mnist.py <br>
> fcn_8_mnist.py <br>
> de_fcn_4_mnist.py <br>
> fcn_4_cifar10.py <br>
> fcn_8_cifar10.py <br>



