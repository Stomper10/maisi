#!/bin/bash

#SBATCH --job-name=brats_recon
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs_ex/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

echo "Validation MAISI BraTS VAE."

python3 maisi_valid_VAE_brats.py \
    --pretrained_vae_path /shared/s1/lab06/wonyoung/maisi/results/EX_maisi_vae_s2_brats/checkpoint-0 \
    --model_config_path configs/config_maisi3d-rflow_brats.json \
    --train_config_path configs/config_maisi_vae_train_brats.json \
    --save_dir /leelabsg/data/ex_MAISI/EX_BraTS_temp/recon \
    --device cuda
