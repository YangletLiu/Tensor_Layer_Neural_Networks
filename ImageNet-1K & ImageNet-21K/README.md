# Classification on ImageNet-1K & ImageNet-21K Dataset

The precision in parentheses is the result in the references.


## Motivation and method  
Most of neural networks consists of convolutional layers and fully connected layers, among which the number of parameters in fully connected layers dominates the total number of parameters of netwroks. For examples,  

(1) AlexNet has 5 convolutional layers and 3 fully connected layers, containing more than 60 million trainable parameters.  It should be noted that the fully connected layers containing more than 58 million parameters, approximately 96 % of the total parameters. 

(2) VGG16 has 13 convolutional layers and 3 fully connected layers, containing more than 138 million trainable parameters. There are more than 123 million parameters in the fully connected layers containing, approximately 89% of the total parameters.
 
The number of parameters in fully connected layer is related to the input size. Taking VGG16 as an example, when we split the ImageNet-1K dataset into 4 sub datasets, the number of parameter in the fully connected layers is reduced from 123 million to 39 million.  
The size of VGG16 is reduced from 527 MB to 207 MB.     

The ResNet network has convolutional layers. It does not have fully connected layers.

This method can reduce the usage of GPU memory during training. 
Taking ResNet34 as an example, using the same settings, our method can reduce the GPU memory consumption during training from 31 GB to approximately 10 GB. Using the same device, The method supports larger batch size during training.
For ResNet34, this method increased the upper limit of batch size from 512 to 2048.  This brings about reduced training time or more flexible hyper-parameter search space.[8]

## ImageNet-1K [1]

### Experimental setting:  

Input size :

    224 x 224 x 3 for baseline
    112 x 112 x 3 for sub4
    56 x 56 x 3 for sub16&sub36

    for sub36, resize the original image to 336 x 336 x 3

Epoch: 100;   Batch size: 256

Optimizer: SGD with momentum as 0.9, and weight-decay as 1e-4;  

Initial lr: 0.01;

lr-scheduler: cosineannealingLR with T_max as 10, and lr_min as 1e-4;

### Numeric results:  


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

" * " is the target accuracy

Experimental results on AlexNet and VGG demonstrate the advantages of this method in model compression and training time.

The 76.1% [6] accuracy for ResNet34 is state-of-the-art (SOTA). A setting of similar accuracy(76.4%)[9] requires 372 hours for training.   Using the same experimental setting of [9], we achieve the 78.29 % accuracy in 281 hours. Our method can be adapted to various experimental setting to achieve SOTA-level accuracy.  For lighter training methods, spectral-ResNet34-sub16 using same experimental setting as the baseline (73.51%).  We also achieve higher accuracy in less training time.

The 75.48 % [5] and 77.15% [5] accuracy for ResNet34 and ResNet50 using the 10-crop testing method.  It takes the center crop and 4 corner crops of images and its horizontal reflection, that we get $ (1 + 4) \times 2 $ images from original image. Networks testing on the 10 images to get 10 result vectors, and average these vectors to get the final result.

10-crop testing will increase the accuracy about 1%~2%, but increase inference cost by 10 times. Most works does not use it in their experiment.  In our experiment, we just use 10-crop testing for resnet50.  spectral-ResNet50-sub4 achieve a relative high accuracy that 78.63 %.  Without 10-crop testing, the accuracy of spectral-ResNet34-sub16 close to baseline, and drop of 1.35 % to accuracy with 10-crop testing.

## ImageNet-21K [2]

ImageNet-21K has 14,197,122 images for 21,841 classes.

Epoch: 100;   

Batch size: 512

Optimizer: SGD with momentum as 0.9, and weight-decay as 1e-4;  

Initial lr: 0.01;

lr-scheduler: cosineannealingLR with T_max as 10, and lr_min as 1e-4;

ImageNet-21K dataset has no official train-validation split. And images in each class is not balanced. 
The whole dataset averages more than 1,000 images per classes. Some classes have fewer than 10 images.
refer to [11], we remove same classes that contain less than 500 images, and allocate 50 images per class for a validation split.
Then the dataset contains 12,358,688 images from 11,221 classes. We will compare our results with [11].

The image size varies in this dataset, e.g., 600 x 426 x 3, 1600 x 1200 x 3, 150 x 113 x 3, 500 x 333 x 3.
We resize this original images to 336 x 336 x 3.

For our spectral method, we split the original dataset into 36 sub-datasets in spectral domain as follows

1. 分割大小为 336 x 336 x 3 的原始图像. 将图像分为多个 6 * 6 大小的像素块, 把不同像素块中位置相同的像素组合，得到新的子图像. 
一共可以得到 6 x 6 份子图像, 每份子图像的大小为 56 x 56 x 3. 数据由 336 x 336 x 3 变为 56 x 56 x 3 x 36.
2. 对数据张量沿最后一维进行 DCT 变换. 有 56 x 56 x 3 条向量分别进行 DCT 变换, DCT 变换对应的向量长度为 36 .在 spectral domain 下得到不同频段的36份子图像.
3. 我们把每份子图像单独存储, 最终得到 36 份子数据集. 该数据集由 56 x 56 x 3 大小的 spectral domain 图像组成.

In disk, images in ImageNet-21K are stored in "JPEG" type. 
The original dataset has 1.1 TB size in disk, and this sub datasets has different size, e.g. 45 GB for sub0, 79 GB for sub1, 64 GB for sub2.

During training, images were loaded from disk to CPU memory, then be transferred from CPU memory to GPU memory.

Images in CPU or GPU memory is a tensor with size of H x W x C, that is the height, width, and channel of the image.
Each pixel is int type or float type, that both have 4 bytes size.

For spectral-ResNet-50-sub36 experiment, we take 256 images for a batch, that the memory size for storing is :

$$
\frac{ 56 \times 56 \times 3 \times 256 \times 4} {1024 \times 1024} = 9.18 MB
$$

Our device that A100 GPUs are connected to the PCI switch infrastructure over x16 PCI Express Gen 4 buses that provide 31.5 Gb/s each for a total of 252 Gb/s bandwidth.

The peak time to transfer this images from CPU memory to GPU memory is :

$$
\frac{ 9.18 MB / 1024 } {252 Gb/s * 8} = 0.00028s
$$

We load each batch images in pipeline. When a batch images be training, we well preload many other batch data in CPU memory. 
After an iteration, we directly transfer a new batch images from CPU memory to GPU memory for training.

The network only has to wait the time that data transfer from CPU memory to GPU memory, 
In our experiment, this process takes 0.0003 seconds.
The same batch data load in CPU memory from disk, that takes average 0.43 seconds, about 39 % of the training time;


| Network     | Top-1 Acc. | Top-5 Acc.| Model size | Training time |
| ----------- |  -------------| --- | --- | --- |
| ResNet-34 | 40.45 % [11] | | 122.35 MB | >246 h  |
| spectral-ResNet-34-sub36|  40.74 % | 69.05 % | 122.35 MB | 90 h |
| ResNet-50 | 42.2 % [11] | |  171.56 MB | - |
| spectral-ResNet-50-sub36 | 40.74 %  | 69.05 % | 171.56 MB | 33.5 h (8 GPUs) |



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

[11] Ridnik T, Ben-Baruch E, Noy A, et al. ImageNet-21K Pretraining for the Masses. Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).
