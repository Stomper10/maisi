#!/bin/bash

#SBATCH --job-name=EX_maisi_vae_s1_brats
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1      # OPTIMIZATION: torchrun이 프로세스를 관리하므로 ntasks는 1로 설정
#SBATCH --cpus-per-task=32       # OPTIMIZATION: 4개 프로세스가 사용할 총 CPU 코어 수 (GPU당 8개)
#SBATCH --partition=P2
#SBATCH --exclude=b31
#SBATCH --time=0-12:00:00        # 충분한 시간 할당
#SBATCH --mem=200GB
#SBATCH --signal=B:SIGUSR1@180   # 시간 만료 3분 전 신호 전송
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/maisi/logs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate wdm

echo "Starting MAISI VAE BraTS DDP training on 4 GPUs..."

# --- requeue 로직은 기존과 동일 ---
max_restarts=1000
restarts=$(scontrol show job ${SLURM_JOB_ID} -o | awk -F'|' '{print $11}')
function resubmit()
{
    echo "Job time limit approaching. Signaling Python to save and exit..."
    if [[ -n $pid ]]; then kill -SIGTERM $pid; fi
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

# --- 실행 명령어 변경 ---
# OPTIMIZATION: 시간 측정을 위해 time 명령어 사용
# DDP CHANGE: torchrun으로 분산 학습 실행, --run_name으로 실험 이름 명시
time torchrun --nproc_per_node=4 maisi_train_VAE_brats.py \
    --model_config_path configs/config_maisi3d-rflow_brats.json \
    --train_config_path configs/config_maisi_vae_train_brats.json \
    --run_name ${SLURM_JOB_NAME} \
    --cpus_per_task ${SLURM_CPUS_PER_TASK} \
    --resume &
pid=$!

wait $pid

echo "Training completed or terminated by signal."
exit 0