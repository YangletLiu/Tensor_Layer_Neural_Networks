# MNIST & CIFAR-10 Dataset Classification

Training parameters

| Parameters                | value  |
| ------------------------ | ------ |
| `-b`,`--batch_size`      | `128`   |
| `--epochs`               | `100`   |
| `--lr`                   | `0.001`  |
| `--opt`                  | `SGD`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `0` |
| `--device`               | `cpu` |

By default, a four-layer fully connected network is used for training on the MNIST dataset
```shell
python train.py
```

## model
Use the `--model-name` parameter to select the network model
```shell
python train.py --model-name FC8Net
```

## dataset
Use the `--model-name` parameter to select the dataset. Currently, the code will automatically download MNIST and CIFAR-10 datasets,  you can use `--datapath` to choose the path to save the dataset or your downloaded data.
```shell
python train.py --dataset MNIST --datapath ./
```

## spectral method
Use the `--trans` parameter to select the transform domain.
`--subnums` is the number of copies of the dataset to be split;  `--l_idx`, `--r_idx` represent the range of sub-datasets to be trained

When using the spectral method, its need to use the `--split` parameter to select a data split method
```shell
python train.py --trans dct --subnums 16 --l_idx 0 --r_idx 16 --split downsample
```

## optimizer
Use the `--opt` parameter to select the optimizer; 
```shell
python train.py --opt adam --momentum 0.9 --weight-decay 0.01
```
## scheduler
Use the `--scheduler` parameter to select the scheduler.
```shell
python train.py --scheduler steplr --lr-step-size 30 --lr-gamma 0.1
```

## geometric distribution
Use the `--geo-p` parameter to adjust the weight of each sub-network when using spectral method.`0 <= --geo-p <= 1`. 

If `--geo-p` is not set or `--geo-p = 0`, each sub-network has the same weight. 

```shell
python train.py --trans dct --subnums 16 --l_idx 0 --r_idx 16 --geo-p 0.3
```

