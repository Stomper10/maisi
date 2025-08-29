#!/bin/bash

#SBATCH --job-name=E0_UKB_fid
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "MAISI UKB FID eval."

python3 /shared/s1/lab06/wonyoung/maisi/fid_score_ukb.py
