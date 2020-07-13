# Tensor_Neural_Netowrk_for_Parallel_Channel_Speedup
1.Aim at developing a tensor neuralnetwork speeded by parallel channel.Train from FashionMNIST,MNIST and CIFAR-10 datasets.  

2.In order to evaluate our model's performance, we display several different network,including `matrix-fully-connected`,`autoencoder` and `CNN`   

##  File structure
> NeuralNetwork_DP
>> TNN -----------------code for tensor neuralnetwork and test result
>>> tnn-4.py  ---------4-layer tensor neuralnetwork <br>
>>> tnn-8.py  ---------8-layer tensor neuralnetwork <br>

>>  DCT-TNN -------------tensor neuralnetwork implemented by DCT transform in DCT domain (not finished yet)
>>> dct-tnn-4.py  ---------4-layer tensor neuralnetwork <br>

>> Matrix-fullyConnected -----------------code for matrix fully connetcted neuralnetwork and test result
>>>  mnn_4.py ------------4-layer matrix fully connected network <br>
>>>  mnn_8.py ------------8-layer matrix fully connected network <br>

>> Autoencoder
>>> autoencoder_test.py ------------8-layer autoencoder neuralnetwork(including 4-layer encoder and 4-layer decoder)

>> CNN
>>> cnn-2.py  ----------2-layer convolutional neuralnetwork

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
For this project, i have divided the process into two parts:first to reproduce, or improve if possible, the results of the paper `stable tnn`;and second to develop a more efficient and stable tensor network modified by `parallel channel speedup` on the basis of the former one.So far,i have completed most codes of the first part and got some results consequently.For the following second part which i am working on, i apply `DCT`(dct) transform on the input layer to get the `frontal slices parallel channel`and then depend on `multinuclear GPU`(including but not limited to `torch.multiprocessing`,`threading` and `concurrent.futures`) to speed up the  train process.On the output layer,i apply the `inverse DCT`(idct) to get the results back to time domain, calculate the loss,do the back-propagation and optimize the model.

## Tensor Neural Network with t-procuct based on Bcirc
This tNN works fine on FashionMNIST and MNIST,but not that well on CIFAR-10.When i began the first test,it is discouraged to find that the loss after every epoch contains considerable `nan` values so that the backward process is obstructed.Fortunately,i found these `nan` values were caused by overstack because of the ultra-big numbers,i.e the output tensor of the network.The problem was then addressed by reducing the elements of the output tensor to a limited range(0 to 5 is preferred through my test) and my resolvement was to simply do the `division` on the output(the divisor for 4-lay network is 1e6 and 1e10 for 8-layer) 

Tensor network based on bcirc performs well on MNIST and FashionMNIST and reduces the parameters during the process,with the ultimate test accuracy of 97% and 98%.However,compared with traditional matrix fully connected network,the tensor type shows a slightly lower speed of contraction.As for loss,the two type network do not differ from each other significantly.  

### Experiments

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/TNN/MNIST/MNIST-Loss.png)

![](https://github.com/hust512/Homomorphic_CP_Tensor_Dcomposition/raw/master/Tensor_NeuralNetwork/NeuralNetwork_DP/TNN/MNIST/MNIST-Acc.png)
