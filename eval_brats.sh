#!/bin/bash

#SBATCH --job-name=FID_EX_BraTS
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
echo "MAISI BraTS FID eval. (real 10k vs. blur real 10k)"
echo "Total val data: 876 / 4 = 219 per modality"

python3 /shared/s1/lab06/wonyoung/maisi/eval_brats.py \
    --gen_data_dir="/leelabsg/data/ex_MAISI/EX_BraTS/predictions/E0_stoch"
