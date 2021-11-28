#!/bin/bash
#SBATCH -J cycle_mlp_4          # 指定作业名
#SBATCH -o train_cycle_mlp_4.out      # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                        # 作业申请 1 个节点
#SBATCH --cpus-per-task=5           # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:5                # 单个节点使用 8 块 GPU 卡

python -u -m torch.distributed.launch --nproc_per_node=5 --master_port 25198 --use_env /home/yanglet/tensor_zy/tensor_layer/NeuralNetwork_DP/beta/dct_cycle_mlp/main.py --model CycleMLP_B5 --batch-size 1024 --num_workers 16 --data-path /colab_space/imagenet/ --net_idx 4

