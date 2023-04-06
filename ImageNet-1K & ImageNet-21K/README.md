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

| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|AlexNet [3]|63.44 % <br> (59.3 % [3])| 224 MB | 40.8 h |
|spectral-AlexNet-sub4| 63.43 % | 37.73 MBx4 | 20.2 h |
|spectral-AlexNet-sub16| 62.18 % | 37.73 MBx4 | 9.7 h |
|VGG-16 [4]|73.21 % <br> (73.00 % [4])| 527.79 MB | 81.2 h |
|Spectral-VGG-16-sub4| 72.82 % | 207.82 MBx4 | 44.14 h |
|Spectral-VGG-16-sub16| 64.24 % | 128.05 MBx16 | 26.84 h |
|ResNet-34 [5] |73.51 % <br> (75.48 % [5]) <br> (76.1 % [6])| 83.15 MB | 43.66 h |
|spectral-ResNet34-sub4| 78.29% | 83.15 MBx4 | 281 h |
|spectral-ResNet34-sub16| 72.13 %| 83.15 MBx16 | 20.02 h |
|spectral-ResNet34-sub36| 69.83 % | 83.15 MBx36 | 20.02 h |
|ResNet-50| 77.99 % <br> (77.15 % [5]) <br> (80.3 % [7]) |97.69 MB| 43.8 h |
|spectral-ResNet50-sub4 |77.84% | 97.69 MBx4 | 62.6 h |
|spectral-ResNet50-sub4 | - | 97.69 MBx16 | - |


## ImageNet-21K [2]

split to 36 sub-datasets after resize to 336 x 336 x 3

Input size : 56 x 56 x 3;

Epoch: 100;

Batch size: 512;

Optimizer: SGD (0.9 momentum，0.0001 weight-decay);

initial lr: 0.001;

lr-scheduler: stepLR(30 step size，0.1 gamma);

| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|ResNet-34| 40.45% | 122.35 MB | >246h  |
|spectral-ResNet-34-sub36| 35.12% | 122.35 MB | 90h |
|ResNet-50| - | 171.56 MB | - |
|spectral-ResNet-50-sub36| - | 122.35 MB | - |


[1] Russakovsky, O.Deng, J.Su, H.et al. ImageNet Large Scale Visual Recognition Challenge. Int J Comput Vis 115, 211–252 (2015). 

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and F.-F. Li, “ImageNet: A large-scale hierarchical image database,” in IEEE CVPR, 2009, pp. 248–255.

[3] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. International Conference on Neural Information Processing Systems, 2012.

[4] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations (ICLR). 2015: 1-14.

[5] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.

[6] Liu Z, Li S, Wu D, et al. Automix: Unveiling the power of mixup for stronger classifiers. European Conference Computer Vision–ECCV 2022.

[7] Pham H, Le Q. Autodropout: Learning dropout patterns to regularize deep networks. AAAI Conference on Artificial Intelligence. 2021, 35(11): 9351-9359
