## MNIST Dataset

Image size: 28 x 28.  

#Epoch: 100; 300 for tNN in [1].

Batch size: 128; 100 for tNN in [1].

Optimizer: Adam.

Rank: 10.

|Networks|Layers|Test accuracy|Learning rate|Initialization|
|-|-|-|-|-|
|FC-4L |[784, 784, 784, 784, 10]|98.64%|0.001|random
|FC-8L |[784, 784, 784, 784, 784, 784, 784, 784, 10]|98.71%|0.001|random
|FC-4L (low-rank)|[784, 10, 784, 10, 784, 10, 784, 10]| 96.42%|0.001|xavier normal
|FC-8L (low-rank)|[784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10, 784, 10]| 96.46%|0.001|xavier normal
|tNN-4L | [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.90%;<br> <98.0% in [1].|0.01|random
|tNN-8L | [(28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 28, 28), (28, 10, 28)]|97.59% (28/100 epochs);<br> ~ 98.0% in [1].|0.01; <br>0.1 in [1].|random
|tNN-8L-row-7 | [(7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 112, 112), (7, 10, 112)]|97.78% |0.001|random
|tNN-8L-row-14 | [(14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 56, 56), (14, 10, 56)]|97.79% |0.001|random
|tNN-8L-row-16 | [(16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 49, 49), (16, 10, 49)]|97.86% |0.001|random
|FC-8L-subnets-28(downsample)| 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 94.51% |0.001|random
|FC-8L-subnets-28(segmentation)| 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 95.10% |0.001|random
|FC-8L-subnets-28(block)| 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 97.01% |0.001|random
|Spectral-tensor-8L-subnets-7| 7 subnetworks: <br>[112 (28x4), 112, 112, 112, 112, 112, 112, 112, 10] for each subnetwork. | 98.26% |0.001|random
|Spectral-tensor-8L-subnets-14| 14 subnetworks: <br>[56 (28x2), 56, 56, 56, 56, 56, 56, 56, 10] for each subnetwork. | 98.14% |0.001|random
|Spectral-tensor-8L-subnets-16| 16 subnetworks: <br>[49 (7x7)), 49, 49, 49, 49, 49, 49, 49, 10] for each subnetwork. | 98.36% |0.001|random
|Spectral-tensor-4L-subnets-28| 28 subnetworks: <br>[28, 28, 28, 28, 10] for each subnetwork. | 94.53% | 0.001| random
|Spectral-tensor-8L-subnets-28| 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 96.95% |0.001|random
<!-- |Spectral-tensor-8L-subnets-28| 28 subnetworks: <br>[28, 28, 28, 28, 28, 28, 28, 28, 10] for each subnetwork. | 94.26% |0.001|random -->

Files:
> fc_4L_mnist.py <br>
> fc_8L_mnist.py <br>
> fc_4L_lowrank_mnist.py <br>
> fc_8L_lowrank_mnist.py <br>
> tnn_4L_mnist.py <br>
> tnn_8L_mnist.py <br>
> tnn_8L_row_7_mnist.py <br>
> tnn_8L_row_14_mnist.py <br>
> tnn_8L_row_16_mnist.py <br>
> FC_8L_subnets_28_downsample_mnist.py <br>
> FC_8L_subnets_28_segmentation_mnist.py <br>
> FC_8L_subnets_28_block_mnist.py <br>
> spectral_tensor_8L_subnets_7_mnist.py <br>
> spectral_tensor_8L_subnets_14_mnist.py <br>
> spectral_tensor_8L_subnets_16_mnist.py <br>
> spectral_tensor_4L_subnets_28_mnist.py <br>
> spectral_tensor_8L_subnets_28_downsample_mnist.py <br>
> faderated_fc_3_nodes.py <br>
> faderated_spectral_tensor_8L_subnets_28_mnist.py <br>
<!-- > spectral_tensor_8L_subnets_28_mnist.py <br> -->

tNN for row-_x_ images: reorganize each image into a matrix with a row of size _x_ and train the corresponding tNN.


**Our spectral tensor networks with _x_ subnetworks**: 

1). Preprocess training dataset: reorganize each image into a matrix with a row of size _x_, perform DCT on the data along the row-dimension (size _x_), and split the training dataset into _x_ subsets corresponding to _x_ spectrals (for each image, each spectral has a vector); 

2). Train _x_ subnetworks (4-layer and 8-layer FC, respectively) with training dataset: the _x_ spectral data as **input** and the corresponding labels as **output**;

3). Obtain the trained _x_ subnetworks and corresponding loss values; 

4). In the testing phase, use the loss values to set weights as 1/loss; get the _x_ spectrals of a new image and input them into the _x_ trained subnetworks; ensemble the _x_ outputs by weighted sum to obtain the predicted label.


Train 3 FC networks on 3 nodes using faderated learning method:

1). Preprocess training dataset:
- On node 1: the original training dataset.
- On node 2 and node 3: 
  * Downsample each image into 28 images: divide each image into blocks of size 4 x 7; the first elements of all blocks are organized into the first downsampled image, the second elements of all blocks are organized into the second downsampled image, ... . Stack the 28 downsampled images, perform DCT on the data along the stacking-dimension (size 28) to obtain the transformed data.
  * For node 2: Set the last 14 spectrals of the transformed data to 0, perform inverse DCT on the data along the stacking-demension (size 28) and inverse-downsample the data to original size, namely 28 x 28.

  * For node 3: Set the first 14 spectrals of the transformed data to 0, perform inverse DCT on the data along the stacking-demension (size 28) and inverse-downsample the data to original size, namely 28 x 28.

2). Train the 3 networks on the 3 nodes in synchronous parallel: Take the average of the 3 gradients computed on the 3 nodes; update the 3 networks with the average gradient using Adam optimizers.


**Our spectral tensor networks with 28 subnetworks using faderated learning method**: 
- Train spectral tensor networks with all 28 subnetworks on node 1.
- Train spectral tensor networks with the last 14 subnetworks on node2.
- Train spectral tensor networks with the last 14 subnetworks on node3.

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

|Networks|Layers |Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|CNN-4-layer|[(Conv, ReLU, MaxPool), (Conv, ReLU, Dropout, MaxPool), (Conv, ReLU, MaxPool), (Dropout, Linear)] | 99.44% | 0.01 | random 
|CNN-4-layer|[(Conv, ReLU), (Conv, ReLU), (Conv, ReLU), (Conv, ReLU, Dropout, MaxPool), (Conv, ReLU), (Conv, ReLU), (Conv, ReLU, MaxPool), (Dropout, Linear)] | 99.47% |  0.01 | random 

Files:
> cnn_4L_mnist.py <br>
> cnn_8L_mnist.py <br>


## CIFAR 10 Dataset
Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Rank: 16.

Optimizer: SGD with momentum = 0.9.

|Network|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|FC-4-layer|[3072, 3072, 3072, 3072, 10]|59.40%|0.01|random
|FC-8-layer|[3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 10]|59.19%|0.01|random
|FC-4-layer (low-rank)|[3072, 16, 3072, 16, 3072, 16, 3072, 10]|51.25%(need to be tuned)|0.01|xavier normal
|FC-8-layer (low-rank)|[3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 16, 3072, 10]|48.33%(need to be tuned)|0.0001|xavier normal

Files:
> fc_4L_cifar10.py <br>
> fc_8L_cifar10.py <br>
> fc_4_lowrank_cifar10.py <br>
> fc_8L_lowrank_cifar10.py <br>

- - - 

Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Optimizer: SGD with momentum = 0.9 for CNN; Adam for spectral convolutional tensor network.

|Network|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|CNN-4-layer|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU, MaxPool, Dropout), (Dropout, Linear)] | 87.04% | 0.05 | random
|CNN-8-layer|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 92.07% | 0.01 | random
|Spectral-convolutional-tensor-9-layer-subnets-2|2 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.37% | 0.001 | random
|Spectral-convolutional-tensor-9-layer-subnets-4|4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 88.23% | 0.001 | random
|Spectral-convolutional-tensor-9-layer-subnets-8|8 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 82.03% | 0.001 | random

Files:
> cnn_4L_cifar10.py <br>
> cnn_8L_cifar10.py <br>
> spectral_conv_tensor_9L_subnets_2_cifar10.py <br>
> spectral_conv_tensor_9L_subnets_4_cifar10.py <br>
> spectral_conv_tensor_9L_subnets_8_cifar10.py <br>

**Our spectral convolutional tensor networks with _x_ subnetworks**: 

1). Preprocess training dataset: reorganize each image into an image with a row of size _x_, perform DCT on the data along the row-dimension (size _x_), and split the training dataset into _x_ subsets corresponding to _x_ spectrals (for each image, each spectral has a tensor);

2). Train _x_ subnetworks (9-layer CNN) with training dataset: the _x_ spectral data as **input** and the corresponding labels as **output**;

3). Obtain the trained _x_ subnetworks and corresponding loss values;

4). In the testing phase, use the loss values to set weights as 1/loss; get the _x_ spectrals of a new image and input them into the _x_ trained subnetworks; fuse the _x_ outputs by weighted sum to obtain the predicted label.

- - -

Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Optimizer: Adam.

|Network|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|CNN-9-layer-subnets-2|[(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] | 90.66% (90/300 epochs)| 0.001 | random
|CNN-10-layer-subnets-4|4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, Dropout), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 88.29% (66/300 epochs) | 0.001 | random
|CNN-10-layer-subnets-8|8 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, Dropout), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 79.81% (43/300 epochs) | 0.001 | random
|Spectral-convolutional-tensor-9-layer-subnets-2|2 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 91.37%| 0.001 | random
|Spectral-convolutional-tensor-10-layer-subnets-4|4 subnetworks: [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, Dropout), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 88.82% | 0.001 | random
|Spectral-convolutional-tensor-10-layer-subnets-8|8 subnetworks: [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, Dropout), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 83.03% | 0.001 | random

Files:
> cnn_9L_subnets_2_downsample_cifar10.py <br>
> cnn_10L_subnets_4_downsample_cifar10.py <br>
> cnn_10L_subnets_8_downsample_cifar10.py| <br>
> spectral_conv_tensor_9L_subnets_2_downsample_cifar10.py <br>
> spectral_conv_tensor_10L_subnets_4_downsample_cifar10.py <br>
> spectral_conv_tensor_10L_subnets_8_downsample_cifar10.py <br>

**CNN / Spectral convolutional tensor networks with _x_ subnetworks using downsampled data**: 

1). Preprocess training dataset: 
- Downsample each image into _x_ images: divide each image into blocks of size _bh_ x _bw_, where _bh_ x _bw_ = _x_; the first elements of all blocks are organized into the first downsampled image, the second elements of all blocks are organized into the second downsampled image, ...
- * For CNN: split the training dataset into _x_ subsets.
  * For spectral convolutional tensor networks: stack the _x_ downsampled images, perform DCT on the data along the stacking-dimension (size _x_), and split the training dataset into _x_ subsets corresponding to _x_ spectrals (for each image, each spectral has a tensor);

2). Train _x_ subnetworks (CNN) with training dataset: the _x_ data as **input** and the corresponding labels as **output**;

3). Obtain the trained _x_ subnetworks and corresponding loss values;

4). In the testing phase, use the loss values to set weights as 1/loss; get the _x_ downsampled images (or spectrals) of a new image and input them into the _x_ trained subnetworks; fuse the _x_ outputs by weighted sum to obtain the predicted label.

- - -

Image size: 32 x 32 x 3.

#Epoch: 300.  

Batch size: 128.

Optimizer: SGD with momentum = 0.9 for CNN;

|Network|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|Spectral-convolutional-tensor-8-layer-subnets-3| 3 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.47% | 0.001 | random 

File:
> spectral_conv_tensor_8L_subnets_3_cifar10.py <br>
 
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

|Network|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|Spectral-convolutional-tensor-8-layer-avg-subnets-4| 4 subnetworks: <br> [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork. | 91.93% | 0.001 | random 

File:
> spectral_conv_tensor_8L_avg_subnets_4_cifar10.py| <br>

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


## ImageNet Dataset
Image size: 224 x 224 x 3.

#Epoch: 100.  

Batch size: 1024.

Optimizer: Adam.

|Network|Layers|Test accuracy|Learning rate|Initialization
|-|-|-|-|-|
|Spectral-convolutional-tensor-10-layer-subnets-16|16 subnetworks: [(Conv, BatchNorm(BN), ReLU), (Conv, ReLU, BN, MaxPool), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU), (Conv, BN, ReLU, MaxPool, Dropout), (Conv, BN, ReLU, MaxPool), (Dropout, Linear)] for each subnetwork.| 61.24% | 0.001 | random
|Spectral-CycleMLP-B5 [2]|16 subnetworks: a CycleMLP-B5 for image size of 56x56 for each subnetwork.| ensemble: -; <br> subnetwork-1: 71.27%| adjusted periodically | -


## __fusing weight experiment__

|subnetwork|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|accuracy|__71.256%__|68.024%|59.916%|62.114%|56.562%|62.544%|58.998%|67.292%|59.382%|65.634%|55.604%|54.582%|44.55%|56.476%|55.398%|66.618%|
|train loss|__3.592__|3.762|4.2606|4.1007|4.3645|4.0166|4.2621|3.7906|4.224|3.8776|4.4321|4.4918|4.956|4.369|4.4248|3.8076|

|weight assignments|fusing accuracy|
|-|-|
|average|73.262%|
|1/train_loss|70.59%|
|1/train_loss^2|70.916%|
|geo (geometric distribution), p=0.6| 73.166% |
|geo, 0.5|73.422%|
|geo, 0.4|__73.538%__|
|geo, 0.3|73.382%|
|geo, 0.2|72.986%|

## __fusing with original network__
The accuracy of the pretrained model: 83.23%.

### original network + 16 subnetworks:
|weight assignments|fusing accuracy|
|-|-|
|average|72.396%|
|1/train_loss|-|
|1/train_loss^2|-|
|geo, p=0.9|__83.202%__|
|geo, p=0.8| 83.1% |
|geo, p=0.7| 82.946% |
|geo, p=0.6| 82.636% |
|geo, 0.5|-|
|geo, 0.4|81.3%|
|geo, 0.3|-|
|geo, 0.2|-|

### original network + subnetwork-0:
|weight assignments|fusing accuracy|
|-|-|
|average||
|1/train_loss|-|
|1/train_loss^2|-|
|geo, p=0.9|__83.202%__|
|geo, p=0.8| - |
|geo, p=0.7| - |
|geo, p=0.6| - |
|geo, 0.5|-|
|geo, 0.4|82.546%|
|geo, 0.3|-|
|geo, 0.2|-|

File:
> spectral_conv_tensor_10L_subnets_28_imagenet.py <br>
> dct_cycle_mlp/main.py


![avatar](./figs/imagenet_acc.png)

![avatar](./figs/imagenet_loss.png)


## Reference

[1] Stable Tensor Neural Networks for Rapid Deep Learning (https://arxiv.org/abs/1811.06569). <br>
[2] CycleMLP: A MLP-like Architecture for Dense Prediction (https://arxiv.org/abs/2107.10224).