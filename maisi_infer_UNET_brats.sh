#!/bin/bash

#SBATCH --job-name=E0_BraTS_temp
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
echo "MAISI brats inference."

export JOB_NAME="E0_BraTS_temp"
#export JOB_NAME=$SLURM_JOB_NAME

python3 /shared/s1/lab06/wonyoung/maisi/maisi_infer_UNET_brats.py \
    --env_config="/leelabsg/data/ex_MAISI/$JOB_NAME/environment_maisi_diff_model.json" \
    --train_config="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi_diff_model.json" \
    --model_def="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi.json" \
    --num_gpus=1 \
