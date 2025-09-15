#!/bin/bash

#SBATCH --job-name=ukb_recon
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

echo "Validation MAISI UKB VAE."

python3 maisi_valid_VAE.py \
    --pretrained_vae_path /shared/s1/lab06/wonyoung/maisi/results/EX_maisi_vae_s2_ukb/checkpoint-0 \
    --model_config_path configs/config_maisi3d-rflow.json \
    --train_config_path configs/config_maisi_vae_train.json \
    --save_dir /leelabsg/data/ex_MAISI/EX_UKB_temp/recon \
    --device cuda
