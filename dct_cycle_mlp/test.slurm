#!/bin/bash
#SBATCH -J test_dct_cycle_mlp                  # 指定作业名
#SBATCH -o test_dct_cycle_mlp.out              # 屏幕上的输出文件重定向到 test_dct_cycle_mlp.out
#SBATCH -N 1                                   # 作业申请 1 个节点
#SBATCH --cpus-per-task=1                      # 单任务使用的 CPU 核心数为 1
#SBATCH --gres=gpu:1                           # 单个节点使用 1 块 GPU 卡
#SBATCH --priority=204800                      # 任务优先级

# 加载运行环境
module load anaconda3

# conda初始化bash
conda init bash

# 重新进入conda环境
source activate

# conda激活自定义环境
conda activate tl

# 运行程序
python main.py --model CycleMLP_B5 --batch-size 128 --num_workers 16 --data-path /colab_space/imagenet/