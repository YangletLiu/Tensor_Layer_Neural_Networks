# Classification on ImageNet-1K & ImageNet-21k Dataset

## ImageNet-1K [1]
| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|AlexNet [3]|59.3 %|224 MB| - |
|AlexNet|63.44 %|224 MB| 40.8 h |
|spectral-AlexNet-sub4|63.43 %| 37.73 MBx4 | 20.2 h |
|VGG-16 [4]|73.00 %|527.79 MB| - |
|VGG-16|73.21 %|527.79 MB| 81.2 h |
|Spectral-VGG-16-sub4|72.82 %| 207.82 MBx4 | 44.14 h |
|Spectral-VGG-16-sub16|64.24 %| 128.05 MBx16 | 26.84 h |
|ResNet-34 [5]|75.48 %|83.15 MB| - |
|ResNet-34(SOTA)[6]|76.1 %|83.15 MB| - |
|ResNet-34|73.51 % |83.15 MB | 43.66 h|
|spectral-ResNet34-sub4|74.13% | 83.15 MBx4 | 20.02 h |
|spectral-ResNet34-sub16|70.45 %|83.15 MBx16|20.02 h|
|ResNet-50 [5]|77.15 %|97.69 MB| - |
|ResNet-50(SOTA)[7]|80.3 %|97.69 MB| - |
|ResNet-50|77.99 %|97.69 MB| - |
|spectral-ResNet50-sub4 |77.84% | 97.69 MBx4 | 62.6 h |


## ImageNet-21K [2]
| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|ResNet-34| 40.45% | 122.35 MB | >246h  |
|spectral-ResNet-34| 30.40% | 122.35 MB | 90h |

[1] Russakovsky, O.Deng, J.Su, H.et al. ImageNet Large Scale Visual Recognition Challenge. Int J Comput Vis 115, 211–252 (2015). https://doi.org/10.1007/s11263-015-0816-y

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and F.-F. Li,
“ImageNet: A large-scale hierarchical image database,” in IEEE
CVPR. Ieee, 2009, pp. 248–255.

[3] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. ImageNet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 (NIPS'12). Curran Associates Inc., Red Hook, NY, USA, 1097–1105..

[4] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[C]// International Conference on Learning Representations (ICLR). 2015: 1-14.

[5] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.

[6] Liu Z, Li S, Wu D, et al. Automix: Unveiling the power of mixup for stronger classifiers[C]//Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIV. Cham: Springer Nature Switzerland, 2022: 441-458.

[7] Pham H, Le Q. Autodropout: Learning dropout patterns to regularize deep networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(11): 9351-9359