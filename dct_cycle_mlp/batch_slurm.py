import os


if __name__ == "__main__":
    for i in range(4, 16):
        with open("run.slurm", "w") as fl:
            fl.write("#!/bin/bash\n")
            fl.write("#SBATCH -J cycle_mlp_{}         # 指定作业名\n".format(i))
            fl.write("#SBATCH -o train_cycle_mlp_{}.out      # 屏幕上的输出文件重定向到 test.out\n".format(i))
            fl.write("#SBATCH -N 1                        # 作业申请 1 个节点\n")
            fl.write("#SBATCH --cpus-per-task=5           # 单任务使用的 CPU 核心数为 4\n")
            fl.write("#SBATCH --gres=gpu:5                # 单个节点使用 8 块 GPU 卡\n")

            fl.write("python -u -m torch.distributed.launch --nproc_per_node=5 --master_port 25198 --use_env /home/yanglet/tensor_zy/tensor_layer/NeuralNetwork_DP/beta/dct_cycle_mlp/main.py --model CycleMLP_B5 --batch-size 1024 --num_workers 16 --data-path /colab_space/imagenet/ --net_idx {}\n".format(i))
        os.system("sbatch run.slurm")
