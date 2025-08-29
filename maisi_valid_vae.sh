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
echo "Validation MAISI VAE."
echo ""val_sliding_window_patch_size": [64, 64, 64],"
echo "
    SlidingWindowInferer(
        roi_size=args.val_sliding_window_patch_size,
        sw_batch_size=1,
        progress=False,
        overlap=0.25,
        device=device,
        sw_device=device,
    )
"

python3 maisi_valid_VAE.py \
    --pretrained_vae_path /shared/s1/lab06/wonyoung/maisi/results/E0_maisi_vae_1427968/checkpoint-220000 \
    --model_config_path configs/config_maisi3d-rflow.json \
    --train_config_path configs/config_maisi_vae_train.json \
    --device cuda
} &
wait
exit 0
