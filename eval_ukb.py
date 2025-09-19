import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy import linalg
from collections import OrderedDict
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, Resized, GaussianSmoothd ###

from resnet3D import resnet50
from scripts.transforms import VAE_Transform

parser = ArgumentParser()

def trim_state_dict_name(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

def get_activations_from_dataloader(model, data_loader, dataset_len, args):
    pred_arr = np.empty((dataset_len, 2048)) # args.dims -> 2048
    start_idx = 0
    for i, batch in enumerate(data_loader):
        if i % 10 == 0:
            print(f'\rPropagating batch {i}/{len(data_loader)}', end='', flush=True)
        
        batch_images = batch["image"].float().cuda()
        with torch.no_grad():
            pred = model(batch_images)

        batch_size = pred.shape[0]
        end_idx = start_idx + batch_size
        pred_arr[start_idx:end_idx] = pred.cpu().numpy()
        start_idx = end_idx
        
    print(' done')
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_feature_extractor():
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())
    ckpt = torch.load("/shared/s1/lab06/wonyoung/maisi/weights/v1/pretrain/resnet_50.pth")
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt, strict=False) # conv_seg is new
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model

def add_assigned_class_to_datalist(datalist, classname):
    for item in datalist:
        item["class"] = classname
    return datalist

def calculate_precision_recall(real_features, gen_features, k=3):
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(real_features)
    real_distances, _ = nn_real.kneighbors(real_features)
    real_radii = real_distances[:, k]
    
    nn_gen_to_real = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_features)
    dist_gen_to_real, indices_gen_to_real = nn_gen_to_real.kneighbors(gen_features)
    precision_bools = dist_gen_to_real.flatten() <= real_radii[indices_gen_to_real.flatten()]
    precision = np.mean(precision_bools)
    
    nn_gen = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(gen_features)
    gen_distances, _ = nn_gen.kneighbors(gen_features)
    gen_radii = gen_distances[:, k]
    
    nn_real_to_gen = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gen_features)
    dist_real_to_gen, indices_real_to_gen = nn_real_to_gen.kneighbors(real_features)
    recall_bools = dist_real_to_gen.flatten() <= gen_radii[indices_real_to_gen.flatten()]
    recall = np.mean(recall_bools)
    
    return precision, recall

def calculate_density_coverage(real_features, gen_features, k=3):
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(real_features)
    nn_gen = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(gen_features)
    
    dist_real_to_real, _ = nn_real.kneighbors(real_features)
    dist_real_to_real_k = dist_real_to_real[:, k]
    
    dist_gen_to_gen, _ = nn_gen.kneighbors(gen_features)
    dist_gen_to_gen_k = dist_gen_to_gen[:, k]
    
    nn_real_k = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real_features)
    nn_gen_k = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(gen_features)
    
    dist_gen_to_real, _ = nn_real_k.kneighbors(gen_features)
    dist_gen_to_real_k = dist_gen_to_real[:, k - 1]
    
    dist_real_to_gen, _ = nn_gen_k.kneighbors(real_features)
    dist_real_to_gen_k = dist_real_to_gen[:, k - 1]
    
    density = np.mean(dist_gen_to_real_k < dist_gen_to_gen_k)
    coverage = np.mean(dist_real_to_gen_k < dist_real_to_real_k)
    
    return density, coverage



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_data_dir", type=str, default="/leelabsg/data/ex_MAISI/EX_UKB/predictions/E0_stoch")
    args = parser.parse_args()
    args.batch_size = 16
    start_time = time.time()
    print("--- Step 1: Extracting features from all real and generated images ---")
    
    # Real 데이터 준비
    real_label_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv" ### valid.csv - real vs real
    real_data_dir = "/leelabsg/data/20252_unzip/"
    real_label_df = pd.read_csv(real_label_dir)[:10000] # [:10000] - real vs real
    real_files_list = [{"image": os.path.join(real_data_dir, image_name)} for image_name in real_label_df["rel_path"]]
    real_files = add_assigned_class_to_datalist(real_files_list, "mri")
    
    # Generated 데이터 준비
    # gen_data_dir = args.gen_data_dir
    # gen_files_list = [{"image": os.path.join(gen_data_dir, f)} for f in os.listdir(gen_data_dir) if "nii.gz" in f]
    # gen_files = add_assigned_class_to_datalist(gen_files_list, "mri")
    ### delete
    gen_label_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv"
    gen_data_dir = "/leelabsg/data/20252_unzip/"
    gen_label_df = pd.read_csv(gen_label_dir)[10000:20000]
    gen_files_list = [{"image": os.path.join(gen_data_dir, image_name)} for image_name in gen_label_df["rel_path"]]
    gen_files = add_assigned_class_to_datalist(gen_files_list, "mri")
    ### delete

    print(f"Found {len(real_files_list)} real images and {len(gen_files_list)} generated images.")

    # 공통 Transform 정의
    transform = Compose([
        VAE_Transform(is_train=False, random_aug=False, k=4, output_dtype=torch.float16, image_keys=["image"]),
        Resized(keys=["image"], spatial_size=(182, 218, 182), mode="trilinear")
    ])
    ### delete
    transform_blur = Compose([
        VAE_Transform(is_train=False, random_aug=False, k=4, output_dtype=torch.float16, image_keys=["image"]),
        GaussianSmoothd(keys=["image"], sigma=1.0),
        Resized(keys=["image"], spatial_size=(182, 218, 182), mode="trilinear")
    ])
    ###
    
    feature_extractor = get_feature_extractor()

    # Real 데이터 특징 추출
    dataset_real = CacheDataset(data=real_files, transform=transform, cache_rate=0.0, num_workers=4)
    dataloader_real = DataLoader(dataset_real, batch_size=args.batch_size, num_workers=4, shuffle=False)
    act_real_full = get_activations_from_dataloader(feature_extractor, dataloader_real, len(dataset_real), args)

    # Generated 데이터 특징 추출
    dataset_gen = CacheDataset(data=gen_files, transform=transform_blur, cache_rate=0.0, num_workers=4) ### transform
    dataloader_gen = DataLoader(dataset_gen, batch_size=args.batch_size, num_workers=4, shuffle=False)
    act_fake_full = get_activations_from_dataloader(feature_extractor, dataloader_gen, len(dataset_gen), args)

    # --- 2. 부트스트래핑을 위한 시드 및 설정 정의 ---
    print("\n--- Step 2: Starting bootstrapping process ---")
    SEEDS = [21, 42, 111, 123, 450, 555, 654, 777, 984, 1000,] ###
             #2011, 2024, 3000, 3456, 4500, 5000, 6789, 8888, 9854, 9999] ###
    num_available = min(len(act_real_full), len(act_fake_full))
    num_samples_to_use = int(num_available * 0.9)
    print(f"Number of samples for each bootstrap iteration: {num_samples_to_use}")

    # 각 메트릭 결과를 저장할 리스트 초기화
    fid_scores, precision_scores, recall_scores, density_scores, coverage_scores = [], [], [], [], []
    
    # --- 3. 시드 기반 부트스트래핑 루프 실행 ---
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Iteration {i+1}/{len(SEEDS)} (Seed: {seed}) ---")
        np.random.seed(seed) # 현재 반복에 대한 시드 설정
        
        # Real과 Fake에서 동일한 개수만큼 랜덤 샘플링
        real_indices = np.random.choice(len(act_real_full), num_samples_to_use, replace=False)
        fake_indices = np.random.choice(len(act_fake_full), num_samples_to_use, replace=False)
        
        act_real_subset = act_real_full[real_indices]
        act_fake_subset = act_fake_full[fake_indices]

        # FID 계산
        m_real, s_real = post_process(act_real_subset)
        m_fake, s_fake = post_process(act_fake_subset)
        fid_value = calculate_frechet_distance(m_real, s_real, m_fake, s_fake)
        fid_scores.append(fid_value)
        print(f"FID: {fid_value:.8f}")

        # Precision & Recall 계산
        precision, recall = calculate_precision_recall(act_real_subset, act_fake_subset, k=3)
        precision_scores.append(precision)
        recall_scores.append(recall)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Density & Coverage 계산
        density, coverage = calculate_density_coverage(act_real_subset, act_fake_subset, k=3)
        density_scores.append(density)
        coverage_scores.append(coverage)
        print(f"Density: {density:.4f}, Coverage: {coverage:.4f}")

    # --- 4. 최종 결과 리포팅 (평균 및 표준편차) ---
    print("\n--- Final Results (Mean ± Std. Dev. over 10 runs) ---")
    print(f"FID:                 {np.mean(fid_scores):.8f} ± {np.std(fid_scores):.8f}")
    print("----------------------------------------------------")
    print(f"Precision:           {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:              {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print("----------------------------------------------------")
    print(f"Density (Precision): {np.mean(density_scores):.4f} ± {np.std(density_scores):.4f}")
    print(f"Coverage (Recall):   {np.mean(coverage_scores):.4f} ± {np.std(coverage_scores):.4f}")
    print("----------------------------------------------------")
    print(f"Total time: {(time.time()-start_time)/60:.2f} minutes.")