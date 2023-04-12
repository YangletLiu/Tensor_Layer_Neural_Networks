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
|spectral-ResNet50-sub4 | 77.84% | 97.69 MBx4 | 62.6 h |
|spectral-ResNet50-sub16 | (*77.00 %) | 97.69 MBx16 | - |

Experimental results on AlexNet and VGG demonstrate the advantages of this method in terms of model compression and training time.

The architecture of AlexNet is composed of 5 convolutional layers and 3 fully connected layers, containing a total of approximately 60 million trainable parameters.
Notably, the fully connected layers contribute significantly to the overall parameter count, with over 58 million, representing approximately 96 % of the total. 

VGG16 is composed of 13 convolutional layers and 3 fully connected layers, with a total of approximately 138 million trainable parameters. 
The parameter of fully connected layers more than 123 million, representing approximately 89% of the total.

The parameter count of a fully connected layer is closely related to the size of the feature maps that are input to that layer. 
Our proposed method can effectively reduce the size of feature maps during training, leading to a reduction in the number of parameters in the fully connected layers.

Taking VGG16 as an example, when we split the dataset into 4 spectral domains, the parameter count in the fully connected layers of VGG16 is reduced from 123 million to 39 million. 
This allows us to reduce the size of the model from 527 MB to 207 MB.

The ResNet network is a fully convolutional network, composed of a stack of convolutional layers except for the last layer —— classifier, therefore it cannot be compressed. 
However, this method reduces the usage of GPU memory during training, achieving the effect of model compression.
Taking ResNet34 as an example, using the same settings, our method can reduce the GPU memory consumption during training from 31 GB to approximately 10 GB. 
This will lower the requirements for our training devices. 
On the other hand, using the same device, our method can increase the upper limit of batch size during training of ResNet34 from 512 to a maximum of 2048. 
This brings about reduced training time and a more flexible hyperparameter search space.[8]

The 76.1% [6]accuracy for ResNet34 current is state-of-the-art (SOTA), which requires 372 hours of training time under a setting of similar accuracy(76.4%)[9]. 
Using the same experimental setting, we achieved a result of 78.29 % in 281 hours, demonstrating that our method can be adapted to various experimental setups to achieve SOTA-level accuracy. 
For lighter training methods, we conducted experiments on 16 subnetworks using an experimental setup with the same cost as the baseline (73.51%). 
We were able to achieve higher accuracy in less time with this approach.

(ResNet50 baseline的 77.99 % 与 77.15 %[5] 使用了 10-crop的验证方式, 这种方式不涉及训练技巧，单纯在验证阶段处理，结果提升1.5%左右, 
我们可以使用同样的验证方法, 77.84 % 的结果能够提升到约 79 % 左右，该方法正在编写代码)


## ImageNet-21K [2]

split to 36 sub-datasets after resize to 336 x 336 x 3

The size of individual sub-datasets after splitting has been reduced from 1.3 TB to 27.27 GB.

Input size : 56 x 56 x 3;

Epoch: 100;

Batch size: 512;

Optimizer: SGD (0.9 momentum，0.0001 weight-decay);

initial lr: 0.001;

lr-scheduler: stepLR(30 step size，0.1 gamma);

| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|ResNet-34| 40.45 % | 122.35 MB | >246 h  |
|spectral-ResNet-34-sub36| 38.80 % | 122.35 MB | 90 h |
|ResNet-50| - | 171.56 MB | - |
|spectral-ResNet-50-sub36| 38.80 % | 171.56 MB | 33.5 h (8 GPUs) |

Images are stored in CPU or GPU memory as three-dimensional tensors, where a grayscale image is represented as a two-dimensional matrix. 
Each pixel is represented as an integer of type int or a floating-point number of type float. 
For instance, in the PyTorch framework, once an image is loaded, it is saved in the format of "torch.FloatTensor", where each data point occupies 4 bytes.
For 14 million $224 \times 224 \times 3$ pixel color images , The actual memory size occupied by：

$$
\frac{224 \times 224 \times 3 \times 14000000 \times 4} {1024 \times 1024 \times 1024} = 7,850.64 GB
$$

When divided into 36 parts, the size of each sub-dataset is approximately:

$$
\frac{672.91} {36} = 218.07 GB
$$ 

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
