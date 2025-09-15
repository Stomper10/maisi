#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "MAISI BraTS preprocessing step 1 for Unet."

python3 /shared/s1/lab06/wonyoung/maisi/maisi_embedding1_UNET_brats.py \
    --pretrained_vae_path="/shared/s1/lab06/wonyoung/maisi/results/EX_maisi_vae_s1_brats_temp/checkpoint-10000/model.pt" \
    --model_def_path="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow_brats.json" \
    --train_config_path="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_diff_model_brats.json" \
    --env_config_path="/shared/s1/lab06/wonyoung/maisi/configs/environment_maisi_diff_model.json" \
    --embedding_dir="/leelabsg/data/ex_MAISI" \
    --train_data_dir="/leelabsg/data/BraTS25/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train" \
    --valid_data_dir="/leelabsg/data/BraTS25/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/valid" \
    --run_name="EX_BraTS_temp"
