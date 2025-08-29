#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "MAISI brats preprocessing step for ControlNet."

python3 /shared/s1/lab06/wonyoung/maisi/maisi_embedding_CTRL_brats.py \
    --pretrained_vae_path="/shared/s1/lab06/wonyoung/maisi/results/E0_maisi_vae_brats_1540048/checkpoint-45000/model.pt" \
    --pretrained_diff_path="/leelabsg/data/ex_MAISI/E0_BraTS_temp/models/diff_unet_ckpt.pt" \
    --model_def_path="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow_brats.json" \
    --train_config_path="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_controlnet_train.json" \
    --env_config_path="/shared/s1/lab06/wonyoung/maisi/configs/environment_maisi_controlnet_train.json" \
    --label_dir="/leelabsg/data/BraTS25/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train" \
    --embedding_dir="/leelabsg/data/ex_MAISI" \
    --run_name="E0_BraTS_temp"
