#!/bin/bash
#SBATCH -J spectral-resnet152x4-leb05-sub0        # 指定作业名
#SBATCH -o spectral-resnet152x4-leb05-sub0.out     # 屏幕上的输出文件重定向到 ***.out
#SBATCH -N 1                        # 作业申请 1 个节点
#SBATCH --cpus-per-task=8           # 单任务使用的 CPU 核心数为
#SBATCH --gres=gpu:2                # 单个节点使用 N 块 GPU 卡

python -u -m train_for_spectral --name cifar10_`date +%F_%H%M%S` --model BiT-M-R152x4 --model-path /colab_space/yanglet/model_weight/ --logdir /colab_space/yanglet/bit_logs --dataset cifar10 --datadir /colab_space/yanglet/dataset --workers 8 --batch_split 4 --idx 0 --label_smoothing 0.05
