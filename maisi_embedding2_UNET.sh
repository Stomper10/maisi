#!/bin/bash

#SBATCH --job-name=maisi_ukb
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=32
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "MAISI UKB preprocessing step 2 for Unet."

export JOB_NAME=${SLURM_JOB_NAME}

python3 /shared/s1/lab06/wonyoung/maisi/maisi_embedding2_UNET.py \
    --env_config="/leelabsg/data/ex_MAISI/$JOB_NAME/environment_maisi_diff_model.json" \
    --train_config="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi_diff_model.json" \
    --model_def="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi.json" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv" \
    --num_gpus=1 \
