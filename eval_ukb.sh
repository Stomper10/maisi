#!/bin/bash

#SBATCH --job-name=FID_EX_UKB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs_ex/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "MAISI UKB FID eval. (real 10k vs. blur real 10k)"
echo "Total val data: 2,526"

python3 /shared/s1/lab06/wonyoung/maisi/eval_ukb.py \
    --gen_data_dir="/leelabsg/data/ex_MAISI/EX_UKB/predictions/E0_stoch"
