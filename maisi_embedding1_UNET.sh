#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "MAISI preprocessing step 1 for Unet."

python3 /shared/s1/lab06/wonyoung/maisi/maisi_embedding1_UNET.py \
    --pretrained_vae_path="/shared/s1/lab06/wonyoung/maisi/results/E0_maisi_vae_1427968/checkpoint-220000/model.pt" \
    --model_def_path="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow.json" \
    --train_config_path="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_diff_model.json" \
    --env_config_path="/shared/s1/lab06/wonyoung/maisi/configs/environment_maisi_diff_model.json" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv" \
    --embedding_dir="/leelabsg/data/ex_MAISI" \
    --data_dir="/leelabsg/data/20252_unzip" \
    --run_name="E0_UKB"
