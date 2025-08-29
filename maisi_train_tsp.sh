#!/bin/bash

#SBATCH --job-name=E2_maisi_vae
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P3
#SBATCH --exclude=c00
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

max_restarts=100
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
echo "without patch inference while fully utilizing TSP. (num_splits)"

python3 maisi_train_VAE_tsp.py \
    --model_config_path configs/config_maisi3d-rflow_tsp.json \
    --train_config_path configs/config_maisi_vae_train_tsp.json \
    --resume \
    --device cuda
} &
wait
exit 0
