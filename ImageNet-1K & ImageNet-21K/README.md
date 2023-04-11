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

使用 AlexNet 与 VGG 的实验可以表现出该方法在模型压缩与训练时间上的优势。

（VGG的实验结果较久未更新，预计后面可以在 sub4 与 sub16 上均达到或超越目前的baseline，sub4目标为 75 %， sub16目标为73 %）

ResNet 网络为全卷积网络，除最后的分类器外整个网络由卷积层堆叠而成，因此没有压缩效果。但该方法降低了训练期间的显存使用，实现模型压缩想要达到的效果。
以 ResNet34 为例，同样设置下，我们的方法可以把训练期间的显存占用从31GB降低为约10GB，这将降低我们对训练设备的要求。
另一方面，同样设备下，我们的方法可以把 ResNet34 训练期间的 batch size 上限从 512 最高提高为2048。这将带了更训练时间的降低和更灵活的超参数搜索空间[8]

ResNet34 SOTA精度为76.1% [6]，在类似精度(76.4 %)的实验设置上，需要372小时训练时间[9]，我们使用相同实验设置，在 281 小时得到了78.29 % 的结果，表明我们的方法可以和各种类型实验方法适配以达到 SOTA 级别的精度。在 16 个子网络的实验中采用和baseline 73.51% 同样成本的实验方案，也能够在更快的时间上得到了更高的精度。

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
|spectral-ResNet-34-sub36| 38.12 % | 122.35 MB | 90 h |
|ResNet-50| - | 171.56 MB | - |
|spectral-ResNet-50-sub36| 38.80 % | 171.56 MB | 33.5 h (8 GPUs) |

图像在内存或显存中以三维张量形式存储，每个像素点为一个 int 类型整数或 float类型的浮点数。14 m 张 $ 224 * 224 * 3 $ 的图片，在内存中实际占用大小为：

$$

224 * 224 * 3 * 14000000 * 4 / (1024 / 1024 / 1024) = 7,850.64 GB 

$$

分成36份时，单个子数据集大小约为:

$$

672.91 / 36 = 218.07 GB

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
