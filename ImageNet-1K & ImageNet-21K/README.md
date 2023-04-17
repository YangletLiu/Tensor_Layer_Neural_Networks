# Classification on ImageNet-1K & ImageNet-21K Dataset

The precision in parentheses is the result in the references.

## ImageNet-1K [1]

Experimental parameters:

Input size :

    224 x 224 x 3 for baseline
    112 x 112 x 3 for sub4
    56 x 56 x 3 for sub16&sub36

    for sub36, resize the original image to 336 x 336 x3

Epoch: 100;   Batch size: 256

Optimizer: SGD (0.9 momentum，0.0001 weight-decay);

Initial lr: 0.01;

lr-scheduler: cosineannealingLR (100 T_max，0.0001 lr_min);

The precision in brackets is the result of the model in the reference paper

10-crop testing for ResNet34 and ResNet50 in original paper.

10-crop testing: crops 10 different regions from an image, then makes predictions for each region, and finally averages the predictions

" * " is the target accuracy

| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|AlexNet [3]|63.44 % <br> (59.3 % [3])| 224 MB | 40.8 h |
|Spectral-AlexNet-sub4| 63.43 % | 61.04 MB x 4 | 20.2 h |
|spectral-AlexNet-sub16| 62.18 % | 37.73 MB x 16 | 9.7 h |
|VGG-16 [4]|73.21 % <br> (73.00 % [4])| 527.79 MB | 81.2 h |
|Spectral-VGG-16-sub4| 72.82 % (*75.00 %) | 207.82 MBx4 | 44.14 h |
|Spectral-VGG-16-sub16| 64.24 % (*73.00 %)| 128.05 MBx16 | 26.84 h |
|ResNet-34 [5] |73.51 % <br> (75.48 % [5]) <br> (76.1 % [6])| 83.15 MB | 43.66 h <br> (372 h for 76.1 %)|
|spectral-ResNet34-sub4| 78.29% | 83.15 MBx4 | 281 h |
|spectral-ResNet34-sub16| 74.13 %| 83.15 MBx16 | 20.02 h |
|spectral-ResNet34-sub36| 69.83 % | 83.15 MBx36 | 20.02 h |
|ResNet-50| 77.99 % <br> (77.15 % [5]) <br> (80.3 % [7]) |97.69 MB| 43.8 h |
|spectral-ResNet50-sub4 | 78.63 % | 97.69 MBx4 | 62.6 h |
|spectral-ResNet50-sub16 | (*77.00 %) | 97.69 MBx16 | - |

Experimental results on AlexNet and VGG demonstrate the advantages of this method in model compression and training time.

AlexNet has 5 convolutional layers and 3 fully connected layers, containing more than 60 million trainable parameters.
the fully connected layers containing more than 58 million parameters, approximately 96 % of the total parameters. 

VGG16 has 13 convolutional layers and 3 fully connected layers, containing more than 138 million trainable parameters. 
the fully connected layers containing more than 123 million parameters, approximately 89% of the total parameters.

The number of parameters of fully connected layer is related to the size of the feature maps. 
Our method can reduce the size of feature maps during training.
Therefore we can reduce the number of parameters in the fully connected layers.

Taking VGG16 as an example, when we split the dataset into 4 sub datasets, the number of parameter in the fully connected layers is reduced from 123 million to 39 million. 
The size of VGG16 is reduced from 527 MB to 207 MB.

The ResNet network is a fully convolutional network. It does not have fully connected layers that can be compressed.
This method can reduce the usage of GPU memory during training, achieving the effect of model compression.
Taking ResNet34 as an example, using the same settings, our method can reduce the GPU memory consumption during training from 31 GB to approximately 10 GB.
The method supports training these networks on devices with less memory.

Using the same device, The method supports larger batch size during training.
For ResNet34, this method increased the upper limit of batch size from 512 to 2048. 
This brings about reduced training time or more flexible hyper-parameter search space.[8]

The 76.1% [6]accuracy for ResNet34 is state-of-the-art (SOTA). A setting of similar accuracy(76.4%)[9] requires 372 hours for training. 
Using the same experimental setting of [9], we achieve the 78.29 % accuracy in 281 hours. Our method can be adapted to various experimental setting to achieve SOTA-level accuracy. 
For lighter training methods, spectral-ResNet34-sub16 using same experimental setting as the baseline (73.51%). 
We also achieve higher accuracy in less training time.

The 75.48 % [5] and 77.15 % [5] accuracy for ResNet34 and ResNet50 using the 10-crop testing method. 
It takes the center crop and 4 corner crops of images and its horizontal reflection.
networks testing on the 10 parts images to get 10 result vectors, and average these vectors to get the final result.

10-crop testing will increase the accuracy about 1%~2%, but increase inference cost by 10 times. 
Most works does not use it in their experiment. 
In our experiment, we just use 10-crop testing for resnet50. 
spectral-ResNet50-sub4 achieve a relative high accuracy that 78.63 %. 
Without 10-crop testing, the accuracy of spectral-ResNet34-sub16 close to baseline, 
and drop of 1.35 % to accuracy with 10-crop testing.

## ImageNet-21K [2]

ImageNet-21K has 14,197,122 images for 21,841 classes.

We resize this images of different sizes to 336 x 336 x 3.
For spectral method, we split the dataset into 36 sub-datasets.

Input size : 56 x 56 x 3;

Epoch: 100;

Batch size: 512;

Optimizer: SGD (0.9 momentum，0.0001 weight-decay);

initial lr: 0.001;

lr-scheduler: stepLR(30 step size，0.1 gamma);

| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|ResNet-34| 40.45 % | 122.35 MB | >246 h  |
|spectral-ResNet-34-sub36| 40.74 % | 122.35 MB | 90 h |
|ResNet-50| - | 171.56 MB | - |
|spectral-ResNet-50-sub36| 38.80 % | 171.56 MB | 33.5 h (8 GPUs) |

Color image in CPU or GPU memory is a three-dimensional tensors, Grayscale image is a two-dimensional matrix. 
Each pixel is int type or float type. 
For instance, in the PyTorch framework, image is saved as "torch.FloatTensor", that size of each pixel is 4 bytes.
The memory size for storing this dataset after resize is :

$$
\frac{336 \times 336 \times 3 \times 14,197,122 \times 4} {1,024 \times 1,024 \times 1,024} = 17,912.66 GB
$$

It is a huge number for memory capacity.

We split the dataset into 36 sub datasets. The size of each sub-dataset is approximately :

$$
\frac{17,912.66} {36} = 497.57 GB
$$ 

Our device, DGX-A100[10], has 2 TB memory. It is able to store the sub-dataset in memory. 
We can directly transfer images from CPU memory to GPU memory during the training. For each batch images that batch size is 512, this process takes 0.003 seconds.
The same batch data load in CPU memory from disk, that takes average 0.43 seconds, about 39 % of the training time;


You can use these weights to obtain our results：[Weight Link](https://pan.baidu.com/s/1PxdMktuot0MF5OJE0BF0UQ?pwd=wiyq) (To be updated)

[1] Russakovsky, O.Deng, J.Su, H.et al. ImageNet Large Scale Visual Recognition Challenge. Int J Comput Vis 115, 211–252 (2015). 

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and F.-F. Li, “ImageNet: A large-scale hierarchical image database,” in IEEE CVPR, 2009, pp. 248–255.

[3] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. International Conference on Neural Information Processing Systems, 2012.

[4] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations (ICLR). 2015: 1-14.

[5] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.

[6] Liu Z, Li S, Wu D, et al. Automix: Unveiling the power of mixup for stronger classifiers. European Conference Computer Vision–ECCV 2022.

[7] Pham H, Le Q. Autodropout: Learning dropout patterns to regularize deep networks. AAAI Conference on Artificial Intelligence. 2021, 35(11): 9351-9359

[8] Smith S L, Kindermans P J, Ying C, et al. Don't Decay the Learning Rate, Increase the Batch Size. International Conference on Learning Representations.

[9] Wightman R, Touvron H, Jégou H. Resnet strikes back: An improved training procedure in timm. arXiv preprint arXiv:2110.00476, 2021.

[10] J. Choquette et al., “NVIDIA A100 tensor core GPU: Performance and innovation,” IEEE Micro, vol. 41, no. 2, pp. 29–35, 2021.
