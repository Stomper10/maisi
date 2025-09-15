#!/bin/bash

#SBATCH --job-name=E5_maisi_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --exclude=b31
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs_ex/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

max_restarts=1000
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)

function resubmit()
{
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

{
echo "Training MAISI VAE from scratch."

python3 maisi_train_VAE.py \
    --model_config_path configs/config_maisi3d-rflow.json \
    --train_config_path configs/config_maisi_vae_train.json \
    --resume \
    --device cuda \
    --run_name E5_maisi_vae_1580729
} &
wait
exit 0
