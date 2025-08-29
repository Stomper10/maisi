#!/bin/bash

#SBATCH --job-name=E0_BraTS_temp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --exclude=b31
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
restarts=$(scontrol show job ${SLURM_JOB_ID} -o | awk -F'|' '{print $11}')

function resubmit()
{
    echo "Job time limit approaching. Signaling Python to save and exit..."
    if [[ -n $pid ]]; then
        kill -SIGTERM $pid
    fi

    wait $pid
    echo "Python process finished gracefully. Requeuing job..."

    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "Training MAISI brats CTRL from scratch."

export JOB_NAME=$SLURM_JOB_NAME

python3 /shared/s1/lab06/wonyoung/maisi/maisi_train_CTRL_brats.py \
    --env_config="/leelabsg/data/ex_MAISI/$JOB_NAME/environment_maisi_controlnet_train.json" \
    --model_config="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi_controlnet_train.json" \
    --model_def="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi.json" \
    --num_gpus=1 \
    --resume &
pid=$!

wait $pid
exit 0