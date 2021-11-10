#!/bin/bash
#SBATCH -J train_cycle_mlp          # 指定作业名
#SBATCH -o train_cycle_mlp.out      # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                        # 作业申请 4 个节点
#SBATCH --cpus-per-task=4           # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:8                # 单个节点使用 8 块 GPU 卡
#SBATCH --priority=204800           # 任务优先级

# 加载运行环境
module load anaconda3

# conda初始化bash
conda init bash

# 重新进入conda环境
source activate

# conda激活自定义环境
conda activate tl

# 运行程序
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 25198 --use_env /home/yanglet/tensor_zy/tensor_layer/NeuralNetwork_DP/beta/dct_cycle_mlp/main.py --model CycleMLP_B5 --batch-size 128 --num_workers 16 --data-path /colab_space/imagenet/