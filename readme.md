# Spectral Tensor Neural Netowrks and its Parallel Implementations

## MNIST Task
<<<<<<< HEAD
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization metod|Optimizer
|-|-|-|-|-|-|-|-|
|4-layer FC|[784, 784, 784, 784, 10]|98.40%|100|0.01|64|random|SGD with 0.9 momentum
|8-layer FC|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.56%|100|0.01|64|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|[784, 8, 784, 8, 784, 8, 784, 8, 10]|95.62%|100|0.01|64|low rank matrix decomposition initialization|SGD with 0.9 momentum
|8-layer FC (decomposed)|[784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 10]| **not test yet**|100|0.01|64|low rank matrix decomposition initialization|SGD with 0.9 momentum

## CIFAR 10 Task
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Batch size|Initialization method|Optimizer
=======
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Initialization metod|Optimizer
|-|-|-|-|-|-|-|
|4-layer FC|[784, 784, 784, 784, 10]|98.40%|100|0.01|random|SGD with 0.9 momentum
|8-layer FC|[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.56%|100|0.01|random|SGD with 0.9 momentum
|4-layer FC (decomposed)|[784, 8, 784, 8, 784, 8, 784, 8, 10]|95.62%|100|0.01|low rank matrix decomposition initialization|SGD with 0.9 momentum
|8-layer FC (decomposed)|[784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 784, 8, 10]| **not test yet**|100|0.01|low rank matrix decomposition initialization|SGD with 0.9 momentum

## CIFAR 10 Task
|Network|Network structure|Test accuracy|#Epoch|Learning rate|Initialization method|Optimizer
>>>>>>> f6219561c461ebded04e8ed429f6566151994f0c
|-|-|-|-|-|-|-|
|4-layer FC|[3072, 4096, 2048, 1024, 10]|56.32%|100|1e-4|128|random|Adam
|8-layer FC|[3072, 4096, 4096, 2048, 2048, 1024, 1024, 512, 10]|56.61%|100|1e-3|128|random|Adam
|4-layer FC (decomposed)|[3072, 8, 4096, 8, 2048, 8, 1024, 8, 10]|**not test yet**|
|8-layer FC (decomposed)|[3072, 8, 4096, 8, 4096, 8, 2048, 8, 2048, 8, 1024, 8, 1024, 8, 512, 8, 10]|**not test yet**|


1. A spectral tensor neural network with parallel/distributed implementation. 

On MNIST, FashionMNIST and CIFAR-10 datasets.  

2. To verify the performance, we compare with several neural networks, including `fully-connected layer`, and `Convolutional neural networks (CNN)`.  

One can test and compare with other tensor approaches in [link](https://github.com/hust512/Tensor_Layer_for_Deep_Neural_Network_Compression/tree/master/transform_based_network).

##  File structure
> fcn_4_mnist.py <br>
> fcn_8_mnist.py <br>
> de_fcn_4_mnist.py <br>
> fcn_4_cifar10.py <br>
> fcn_8_cifar10.py <br>

> NeuralNetwork_DP
>> TNN -----------------code for tensor neuralnetwork and test result
>>> tnn-4.py  ---------4-layer tensor neuralnetwork <br>
>>> tnn-8.py  ---------8-layer tensor neuralnetwork <br>

>>  DCT-TNN -------------tensor neuralnetwork implemented by DCT transform in DCT domain (not finished yet)
>>> dct-tnn-4.py  ---------4-layer tensor neuralnetwork <br>


>> Autoencoder
>>> autoencoder_test.py ------------8-layer autoencoder neuralnetwork(including 4-layer encoder and 4-layer decoder)


## Usage
Take `TNN\tnn.py` as example:  

1.Define parameters of `run_all` function.Parameters are as follows:  
  * dataset : choose a dataset from MNIST or FashionMNIST.(CIFAFR-10 for this network is defined in `TNN\CIFAR`)
  * net_layers : 4 or 8 is provided for this module.
  * batch_size : train and test batch size.Defaulted to be 100.
  * lr_rate : learning rate for optimation
  * epochs_num : numbers of iterations for training.

2.Run `run_all` function.Example:  
  ```python
  run_all('MNIST',4,100,0.1,100)
  ```
  This function creates two files:
  * 'tnn_test.csv': stores info for every single train and test epoch
  * 'tnn.jpg' : displays the `accuracy` and `loss` after every epoch for train and test process respectively.
 
##  Method
For this project, we have divided the process into two parts: first to reproduce, or improve if possible, the results of the paper `stable tnn`;and second to develop a more efficient and stable tensor network modified by `parallel channel speedup` on the basis of the former one.So far,i have completed most codes of the first part and got some results consequently.For the following second part which i am working on, i apply `DCT`(dct) transform on the input layer to get the `frontal slices parallel channel`and then depend on `multinuclear GPU`(including but not limited to `torch.multiprocessing`,`threading` and `concurrent.futures`) to speed up the  train process.On the output layer,i apply the `inverse DCT`(idct) to get the results back to time domain, calculate the loss,do the back-propagation and optimize the model.

## Tensor Neural Network with t-procuct based on Bcirc
This tNN works fine on FashionMNIST and MNIST, but not that well on CIFAR-10. 

1. issue 1: the loss after every epoch contains considerable `nan` values so that the backward process is obstructed. These `nan` values were caused by overstack because of the ultra-big numbers, i.e the output tensor of the network. The problem was then addressed by reducing the elements of the output tensor to a limited range (0 to 5 is preferred through my test) and our solution was to simply do the `division` on the output (the divisor for 4-lay network is 1e6 and 1e10 for 8-layer) 

## Tensor Neural Network with t-product based on DCT 'paralel' channel
Further modification should be done both on the code and the network structure to reach the speedup goal.  

The following attempt provide a scratch of our direction.
```python
def forward(self, x):

   dct_x = torch.transpose(dct(torch.transpose(x, 0, 2)), 0, 2)     # do the DCT transform along the third dimension
   
   frontal_slices = []
   for i in range(dct_x.shape[0]):                                  # do the fraontal-slice-wise matrix multiplication
      layer_slice_1 = torch.mm(self.W_1[i, :, :],dct_x[i, :, :]) + self.B_1[i, :, :]
      F.relu(layer_slice_1)
      layer_slice_2 = torch.mm(self.W_2[i, :, :],layer_slice_1) + self.B_2[i, :, :]
      F.relu(layer_slice_2)
      layer_slice_3 = torch.mm(self.W_3[i, :, :],layer_slice_2) + self.B_3[i, :, :]
      F.relu(layer_slice_3)
      layer_slice_4 = torch.mm(self.W_4[i, :, :],layer_slice_3) + self.B_4[i, :, :]
      F.relu(layer_slice_4)
      frontal_slices.append(layer_slice_4)
      
    output = torch.stack(frontal_slices)
    idct_x = torch.transpose(idct(torch.transpose(output,0,2)),0,2) #do the inverse DCT transform 
    return idct_x

```
For rough evaluation, this kind of forward algorithms performs better than original tensor network based on bcirc, with a speedup ratio of nearly `1.17`, which is calculated by comparision of the total running time (100 epochs) of original tNN and the above one. 
***

## Autoencoder
We would like to peruse the decoder part on the multi-spectrum dataset, which is being done now. A reference is ECCV_SD_CASSI and `https://github.com/mengziyi64/SMEM`. Besides, autoencoder and decoder implemented by matrix fully connected are provided to be modified.
***

## Convolutional Neural Network
Integrating the convolutional layers and fully connected layers yields a better than traditional tensor ones, showing both quicker contraction speed as well as high enough accuracy. Subsequently, we apply this kind of sturcture to parallel channels generated by DCT transform to enhance the performance of the tNN.
***

## Experiments
We have trained four types of networks and compared the results on MNIST and CIFAR-10. 

Besides,the simple autoencoder and convolutional neuralnetwork testing results are also included on the bottom. 
***
### Experiments on MNIST

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/MNIST_loss.png)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/MNIST_acc.png)

From the test result graph,we can see that the Tensor network based on bcirc performs well on MNIST and FashionMNIST and reduces the parameters during the process,with the ultimate test accuracy of 97% and 98%.However,compared with traditional matrix fully connected network,the tensor type shows a slightly lower speed of contraction.As for loss,the two type network do not differ from each other significantly.  
***
### Experiments on CIFAR

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/cifar10_loss.PNG)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/cifar10_acc.PNG)

From the test result, we can find that the TNN does not work as well as the matrix one regarding to the accuracy.However,there is an obvious flaw in both mnn-4 and mnn-8 that the loss line meets a regular climb as it decreaces to a lowest value.What i should mention is that, in reference to the paper(stable TNN...),the two TNNs achieve a stable test accuracy of about 47% as the paper does despite that they do not perform as well as the MNN with a final test accuracy of about 57%.
***

### Experiments on MNIST

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/tnn4_mnist_acc_9_8.png)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/log_scale_acc.png)

### Experiments on CIFAR-10

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/tnn4_cifar_acc.png)

The results above are from the latest modified TNN-4, which has a similar parameters-initiation method as the 'nn.Linear' does. Besides,the learning rate was reset as '0.01' instead of '0.1', which contributes to the better performance.
***



