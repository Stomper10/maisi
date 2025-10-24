#!/bin/bash

#SBATCH --job-name=ukb_real_vs_gen-ckpt200k
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

echo "Validation MAISI UKB real_vs_gen"

python3 alleval_ukb.py \
    --pretrained_vae_path /shared/s1/lab06/wonyoung/maisi/weights/ukb/ukb_vae_stage2 \
    --pretrained_unet_path /leelabsg/data/ex_MAISI/maisi_ukb/models/checkpoint-200000 \
    --model_config_path /leelabsg/data/ex_MAISI/maisi_ukb/config_maisi.json \
    --vae_config_path /shared/s1/lab06/wonyoung/maisi/weights/ukb/ukb_vae_stage2/config_maisi_vae_train.json \
    --diff_config_path /leelabsg/data/ex_MAISI/maisi_ukb/config_maisi_diff_model.json \
    --eval_mode real_vs_gen \
    --save_dir /leelabsg/data/ex_MAISI/maisi_ukb \
    --device cuda \
    --base_label_dir /shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv \
    --other_label_dir /shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv \
