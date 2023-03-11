# Classification on MNIST & CIFAR-10 Dataset 

The code will automatically download MNIST and CIFAR-10 datasets

Training parameters

| Parameters               | values  |
| ------------------------ | ------ |
| `-b`,`--batch_size`      | `128`   |
| `--epochs`               | `100`   |
| `--lr`                   | `0.001`  |
| `--opt`                  | `SGD`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `0` |
| `--device`               | `cpu` |



## MNIST Dataset [1]

Image size: 28 x 28.  

Epoch: 100

Batch size: 256

Optimizer: Adam.

Learning rate: 0.001

Rank: 10.

| Networks         | Test accuracy | Model size | Training time|
| ---------------- | ------------- | --- | --- |
| FC-4L            | 98.64%        | 7.10 MB | 1,496s |
| FC-8L            |98.79%        | 16.54 MB | 1,553s |
| t-NN             |97.71% | 0.63 MB | 61,960s |
| Spectral-FC-8L-subnets-4 |  98.88%  | 1.06x4 MB| 1,944s|
| Spectral-FC-8L-subnets-16 |  97.84%  | 0.07x16 MB| 3,470s |
```shell
command :

python train.py --opt adam --model-name FC4Net
python train.py --model-name FC8Net --scheduler steplr --b 256 -j 8 --lr 0.001 --opt adam
python train.py --model-name FC8Net --scheduler steplr --b 256 --lr 0.001 --trans fft --l_idx 0 --r_idx 4 --split downsample --opt adam --filename spectral-fc8l-sub4 --device 0 --geo-p 0.9
```

![img.png](../figs/AccuracyOnMNIST.png)

![img.png](../figs/TrainingLossOnMNIST.png)
## CIFAR 10 Dataset [2]

Image size: 32 x 32 x 3.

### CNN:

Epoch: 300

Batch size: 256

lr: 0.01

### ResNet152x4
Image size : training 160 x 160; validating 128 x 128.

Epoch: 100

Batch size:512

lr: 0.003, multiply by 0.1 every 30 epochs

label smoothing 0.5

opt: SGD with momentum = 0.9

mixup: 0.1

pretrained on ImageNet-21K

| Network     | Test accuracy | Model size | Training time|
| ----------- |  ------------- | --- | --- |
|FC-8L| 61.27% | 252.52 MB | 3353s |
|spectral-FC-sub4| 68.17% | 15.88 MBx4|3999s|
|spectral-FC-sub16| 59.95% | 1.01 MBx16|6639s|
| ResNet152x4 | 99.21% | 3541.64 MB| 15.2h |
| spectral-ResNet152x4-subnets-4| 99.20 %| 3541.64 MBx4 | 17.3 h |
```shell
command :

python train.py --dataset cifar10 --model-name CNN8CIFAR10 --epochs 300 --opt adam
python train.py --dataset cifar10 --model-name CNN10CIFAR10 --epochs 300 --opt adam --trans dct --l_idx 0 --r_idx 4 --split downsample --pretrain ./CNN8CIFAR10.pth
```
![img.png](../figs/AccuracyOnCIFAR10.png)

![img.png](../figs/TrainingLossonCIFAR10.png)
[1] L. Deng, “The MNIST database of handwritten digit images for machine learning research,” IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 141–142, 2012.

[2] A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” Master’s thesis, University of Tront, 2009.