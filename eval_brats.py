import os
import time
import glob
import argparse
import numpy as np
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
    parser.add_argument("--gen_data_dir", type=str, default="/leelabsg/data/ex_MAISI/EX_BraTS/predictions/E0_stoch")
    args = parser.parse_args()
    args.batch_size = 16
    start_time = time.time()

    # <<< 개선점 1: 루프 밖에서 모델과 Transform을 한 번만 로드 >>>
    feature_extractor = get_feature_extractor()
    transform = Compose([
        VAE_Transform(is_train=False, random_aug=False, k=4, output_dtype=torch.float16, image_keys=["image"]),
        Resized(keys=["image"], spatial_size=(240, 240, 155), mode="trilinear")
    ])
    ### delete
    transform_blur = Compose([
        VAE_Transform(is_train=False, random_aug=False, k=4, output_dtype=torch.float16, image_keys=["image"]),
        GaussianSmoothd(keys=["image"], sigma=1.0),
        Resized(keys=["image"], spatial_size=(240, 240, 155), mode="trilinear")
    ])
    ###
    
    MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
    SEEDS = [21, 42, 111, 123, 450, 555, 654, 777, 984, 1000,
             2011, 2024, 3000, 3456, 4500, 5000, 6789, 8888, 9854, 9999]

    # 최종 결과를 저장할 딕셔너리
    final_results = {}

    for mod in MODALITIES:
        print(f"\n{'='*20} Processing Modality: {mod.upper()} {'='*20}")
        
        # --- 1. Modality별 데이터 경로 설정 및 특징 추출 ---
        
        # Real 데이터 준비
        real_data_dir = "/leelabsg/data/BraTS25/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train" ### valid - real vs real
        real_filepaths = sorted(glob.glob(os.path.join(real_data_dir, "*", f"*-{mod}.nii.gz")))
        real_files_list = [{"image": f} for f in real_filepaths[:516]] ### [:516] - real vs real
        real_files = add_assigned_class_to_datalist(real_files_list, "mri")
        
        # <<< 개선점 2: Modality에 맞는 생성 데이터만 필터링 >>>
        # 생성된 파일 이름에 modality 정보가 포함되어 있다고 가정 (예: BraTS-GLI-00000-000-t1n.nii.gz)
        # gen_data_dir = args.gen_data_dir
        # gen_filepaths = sorted(glob.glob(os.path.join(gen_data_dir, f"*-{mod}.nii.gz")))
        # gen_files_list = [{"image": f} for f in gen_filepaths]
        # gen_files = add_assigned_class_to_datalist(gen_files_list, "mri")
        ### delete
        gen_data_dir = "/leelabsg/data/BraTS25/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train" ###
        gen_filepaths = sorted(glob.glob(os.path.join(gen_data_dir, "*", f"*-{mod}.nii.gz")))
        gen_files_list = [{"image": f} for f in gen_filepaths[516:1032]] ###
        gen_files = add_assigned_class_to_datalist(gen_files_list, "mri")
        ### delete

        if not real_files_list or not gen_files_list:
            print(f"Data not found for modality {mod}. Skipping.")
            continue
            
        print(f"Found {len(real_files_list)} real images and {len(gen_files_list)} generated images for {mod}.")

        # Real 데이터 특징 추출
        dataset_real = CacheDataset(data=real_files, transform=transform, cache_rate=0.0, num_workers=4)
        dataloader_real = DataLoader(dataset_real, batch_size=args.batch_size, num_workers=4, shuffle=False)
        act_real_full = get_activations_from_dataloader(feature_extractor, dataloader_real, len(dataset_real), args)

        # Generated 데이터 특징 추출
        dataset_gen = CacheDataset(data=gen_files, transform=transform_blur, cache_rate=0.0, num_workers=4) ###
        dataloader_gen = DataLoader(dataset_gen, batch_size=args.batch_size, num_workers=4, shuffle=False)
        act_fake_full = get_activations_from_dataloader(feature_extractor, dataloader_gen, len(dataset_gen), args)

        # --- 2. 부트스트래핑 실행 ---
        print(f"\n--- Starting bootstrapping process for {mod} ---")
        
        # <<< 개선점 3: 샘플 수를 동적으로 계산 (90% 샘플링) >>>
        num_available = min(len(act_real_full), len(act_fake_full))
        num_samples_to_use = int(num_available * 0.9)
        print(f"Number of samples for each bootstrap iteration: {num_samples_to_use} (90% of {num_available})")

        fid_scores, precision_scores, recall_scores, density_scores, coverage_scores = [], [], [], [], []
        
        for i, seed in enumerate(SEEDS):
            np.random.seed(seed)
            real_indices = np.random.choice(len(act_real_full), num_samples_to_use, replace=False)
            fake_indices = np.random.choice(len(act_fake_full), num_samples_to_use, replace=False)
            
            act_real_subset = act_real_full[real_indices]
            act_fake_subset = act_fake_full[fake_indices]

            # 모든 메트릭 계산
            m_real, s_real = post_process(act_real_subset)
            m_fake, s_fake = post_process(act_fake_subset)
            fid_scores.append(calculate_frechet_distance(m_real, s_real, m_fake, s_fake))
            precision, recall = calculate_precision_recall(act_real_subset, act_fake_subset, k=3)
            precision_scores.append(precision)
            recall_scores.append(recall)
            density, coverage = calculate_density_coverage(act_real_subset, act_fake_subset, k=3)
            density_scores.append(density)
            coverage_scores.append(coverage)
            
            print(f'\rIteration {i+1}/{len(SEEDS)} done.', end='', flush=True)
        print("\nBootstrapping finished.")

        # --- 3. 해당 Modality 결과 저장 및 출력 ---
        final_results[mod] = {
            "FID": (np.mean(fid_scores), np.std(fid_scores)),
            "Precision": (np.mean(precision_scores), np.std(precision_scores)),
            "Recall": (np.mean(recall_scores), np.std(recall_scores)),
            "Density": (np.mean(density_scores), np.std(density_scores)),
            "Coverage": (np.mean(coverage_scores), np.std(coverage_scores)),
        }

        print(f"\n--- {mod.upper()}: Final Results (Mean ± Std. Dev. over {len(SEEDS)} runs) ---")
        print(f"FID:                 {final_results[mod]['FID'][0]:.4f} ± {final_results[mod]['FID'][1]:.4f}")
        print(f"Precision:           {final_results[mod]['Precision'][0]:.4f} ± {final_results[mod]['Precision'][1]:.4f}")
        print(f"Recall:              {final_results[mod]['Recall'][0]:.4f} ± {final_results[mod]['Recall'][1]:.4f}")
        print(f"Density:             {final_results[mod]['Density'][0]:.4f} ± {final_results[mod]['Density'][1]:.4f}")
        print(f"Coverage:            {final_results[mod]['Coverage'][0]:.4f} ± {final_results[mod]['Coverage'][1]:.4f}")

    # --- 4. 모든 Modality에 대한 최종 결과 요약 출력 ---
    print(f"\n\n{'='*25} FINAL SUMMARY {'='*25}")
    header = f"{'Modality':<10} | {'FID':^18} | {'Precision':^18} | {'Recall':^18} | {'Density':^18} | {'Coverage':^18}"
    print(header)
    print('-' * len(header))
    for mod, metrics in final_results.items():
        fid_str = f"{metrics['FID'][0]:.3f} ± {metrics['FID'][1]:.3f}"
        prec_str = f"{metrics['Precision'][0]:.3f} ± {metrics['Precision'][1]:.3f}"
        rec_str = f"{metrics['Recall'][0]:.3f} ± {metrics['Recall'][1]:.3f}"
        den_str = f"{metrics['Density'][0]:.3f} ± {metrics['Density'][1]:.3f}"
        cov_str = f"{metrics['Coverage'][0]:.3f} ± {metrics['Coverage'][1]:.3f}"
        print(f"{mod.upper():<10} | {fid_str:^18} | {prec_str:^18} | {rec_str:^18} | {den_str:^18} | {cov_str:^18}")
    print(f"{'='*68}")
    print(f"Total time: {(time.time()-start_time)/60:.2f} minutes.")