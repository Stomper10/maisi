#!/bin/bash

#SBATCH --job-name=EX_BraTS_temp
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --partition=P2
#SBATCH --exclude=b31
#SBATCH --time=0-12:00:00
#SBATCH --mem=200GB
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs_ex/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)  # 단일 노드면 127.0.0.1도 OK
export MASTER_PORT=$((10000 + RANDOM % 50000))
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
ulimit -n 1048576

echo "Starting MAISI UNET BraTS DDP training on 4 GPUs..."

# --- requeue 로직은 기존과 동일 ---
max_restarts=1000
restarts=$(scontrol show job ${SLURM_JOB_ID} -o | awk -F'|' '{print $11}')
function resubmit()
{
    echo "Job time limit approaching. Signaling Python to save and exit..."
    if [[ -n $pid ]]; then kill -SIGTERM -- -$pid; fi
    wait $pid
    echo "Python process finished gracefully. Requeuing job..."
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
    else
        echo "Job is over the Maximum restarts limit."
        exit 1
    fi
}
trap 'resubmit' SIGUSR1

export JOB_NAME="EX_BraTS_temp"
set -m
time torchrun --nproc_per_node=4 maisi_train_UNET_brats.py \
    --env_config_path="/leelabsg/data/ex_MAISI/$JOB_NAME/environment_maisi_diff_model.json" \
    --train_config_path="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi_diff_model.json" \
    --model_config_path="/leelabsg/data/ex_MAISI/$JOB_NAME/config_maisi.json" \
    --cpus_per_task=${SLURM_CPUS_PER_TASK} \
    --run_name=$JOB_NAME \
    --resume &
pid=$!
set +m
wait $pid

echo "Training completed or terminated by signal."
exit_code=$?

if [[ $exit_code -ne 0 ]]; then
  echo "Python exited with code $exit_code"
  exit $exit_code
fi

echo "Training completed or terminated by signal."
exit 0