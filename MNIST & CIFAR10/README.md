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



## MNIST Dataset

Image size: 28 x 28.  

Epoch: 100

Batch size: 128

Optimizer: Adam.

Rank: 10.

| Networks         | Test accuracy | Learning rate | remark |
| ---------------- | ------------- | ------------- | -------------- |
| FC-4L            | 98.64%        | 0.001         | -         |
| FC-8L            |98.51%        | 0.001         | -         |
| Spectral-FC-8L-subnets-4 |  98.39%  | 0.001 | 98.53% for sub0 |
```shell
command :

python train.py --opt adam --model-name FC4Net
python train.py --opt adam --model-name FC8Net 
python train.py --opt adam --model-name FC8Net --trans dct --l_idx 0 --r_idx 4  --split downsample
```

## CIFAR 10 Dataset

Image size: 32 x 32 x 3.

Epoch: 300.

Batch size: 128.

| Network     | Test accuracy | Learning rate | opt |
| ----------- |  ------------- | ------------- | -------------- |
| CNN-8-layer |  92.42% | 0.001          | adam        |
```shell
command :

python train.py --dataset cifar10 --model-name CNN8CIFAR10 --epochs 300 --lr 0.01
python train.py --dataset cifar10 --model-name CNN8CIFAR10 --epochs 300
python train.py --dataset cifar10 --model-name CNN8CIFAR10 --epochs 300 --opt adam
```

Image size : training 160 x 160; validating 128 x 128.

Epoch: 100.

Batch size: 512.

Optimizer: SGD with momentum = 0.9

lr: 0.003, multiply by 0.1 every 30 epochs

mixup: 0.1

pretrained on ImageNet-21K

| Network     | Test accuracy | base_lr | opt |
| ----------- | ------------- | -------------- | -------------- |
| resnet-152x4| 99.03% | 0.003 | SGD |

```shell
command :

1. cd ./reference_code/bit
2. CUDA_VISIBLE_DEVICES=1,2,3,4 python -m train --dataset cifar10 --model BiT-M-R152x4 --name cifar10_`date +%F_%H%M%S` --logdir ./bit_logs --batch_split 4
```

| Network     | Test accuracy | Learning rate | opt |
| -----------  | ------------- | ------------- | -------------- |
| Spectral-CNN-9-layer-subnets-4  | 91.68% | 0.001 | adam with steplr scheduler |
| Spectral-CNN-10-layer-subnets-4<br>(pretrained on ImageNet for 5 epochs)  | 93.98% | 0.001 | adam |
```shell
command :

python train.py --dataset cifar10 --model-name CNN9CIFAR10 --epochs 300 --opt adam --scheduler steplr --lr-step-size 30 --lr-gamma 0.1 --trans dct --l_idx 0 --r_idx 4 --split downsample
python train.py --dataset cifar10 --model-name CNN10CIFAR10 --epochs 300 --opt adam --trans dct --l_idx 0 --r_idx 4 --split downsample --pretrain ./CNN8CIFAR10.pth
```
| Network     | Layers                                                       | Test accuracy | Learning rate | opt |
| ----------- | ------------------------------------------------------------ | ------------- | ------------- | -------------- |
| Spectral-resnext101_64x4d-subnets-4<br>(pretrained on ImageNet) | 4 subnetworks: <br> [spectral resnext101_64x4d with 10 num_classes] for each subnetwork. | 98.23% | 0.2 with lr scheduler | SGD |
```shell
cd ./reference_code
```
training:

```shell
command :

python train.py --model resnext101_64x4d --batch-size 512 --lr 0.2 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.0001 --norm-weight-decay 0.0 --mixup-alpha 0.2 --cutmix-alpha 1.0 --idx i --output-dir . \
--train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4 --label-smoothing 0.05

// one times command gets one subnetwork, --idx i to gets i-th subnetwork
```

ensemble:
```shell
command :

python ensemble.py --model-name resnext101_64x4d --l_idx 0 --r_idx 4 --checkpoint-path spectral_resnext101_64x4d_subx.pth
```
| Network     | Test accuracy | Learning rate | opt |
| -----------  | ------------- | ------------- | -------------- |
| Spectral-resnet50-subnets-4+4<br>(4 dct + 4 fft, pretrain on ImageNet)  | 98.20% | 0.2 with lr scheduler | SGD |
```shell
command :

1. python train.py --dataset cifar10 --model-name resnet50 --epochs 300 --trans dct --l_idx 0 --r_idx 4 --split downsample
2. python train.py --dataset cifar10 --model-name resnet50 --epochs 300 --trans fft --l_idx 0 --r_idx 4 --split downsample
3. python ensemble.py --model-name CNN9CIFAR10 --l_idx 0 --r_idx 8 --dct-nets 4 --checkpoint-path spectral_resnet50_subx.pth

```
