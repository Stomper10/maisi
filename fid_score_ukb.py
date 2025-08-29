#!/usr/bin/env python3

import os
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import linalg

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from resnet3D import resnet50
from scripts.transforms import VAE_Transform
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, Resized

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

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

def get_activations_from_dataloader(model, data_loader, args):
    pred_arr = np.empty((args.num_samples, 2048)) ### args.dims
    for i, batch in enumerate(data_loader):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        batch = batch["image"].float().cuda() ###
        with torch.no_grad():
            pred = model(batch)
            #print("### pred.shape", pred.shape, flush=True)

        if i*args.batch_size > pred_arr.shape[0]:
            pred_arr[i*args.batch_size:] = pred.cpu().numpy()
        else:
            pred_arr[i*args.batch_size:(i+1)*args.batch_size] = pred.cpu().numpy()
    print(' done')
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_feature_extractor():
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                   Flatten()) # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load("/shared/s1/lab06/wonyoung/diffusers/wldm/weights/pretrain/resnet_50.pth")
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt) # No conv_seg module in ckpt
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model

def calculate_fid_real(args):
    """Calculates the FID of two paths"""
    #assert os.path.exists("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")
    args.train_label_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv"
    args.data_dir = "/leelabsg/data/20252_unzip/"
    args.batch_size = 2

    model = get_feature_extractor()
    
    train_label_df = pd.read_csv(args.train_label_dir)[1000:2000]
    train_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in train_label_df["rel_path"]]
    datasets = {
        1: {
            "data_name": "T1_brain_to_MNI",
            "train_files": train_files,
            "modality": "mri",
        }
    }
    train_files = {"mri": []}
    # Function to add assigned class to datalist
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    # Process datasets
    for _, dataset in datasets.items():
        train_files_i = dataset["train_files"]
        print(f"{dataset['data_name']}: number of training data is {len(train_files_i)}.")
        # attach modality to each file
        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
    
    for modality in train_files.keys():
        print(f"Total number of training data for {modality} is {len(train_files[modality])}.")

    train_files_combined = train_files["mri"]
    train_transform = VAE_Transform(
        is_train=False,
        random_aug=False,  # whether apply random data augmentation for training
        k=4,  # patches should be divisible by k
        patch_size=[182, 218, 182],
        val_patch_size=None,
        output_dtype=torch.float32,  # final data type
        spacing_type="original",
        spacing=None,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    train_transform = Compose([
        train_transform,
        Resized(
            keys=["image"],
            spatial_size=(182,218,182),     # 원하는 해상도 (tuple 또는 list)
            size_mode="all",
            allow_missing_keys=True
        )
    ])
    dataset_train = CacheDataset(data=train_files_combined, transform=train_transform, cache_rate=0.0, num_workers=8)
    data_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, shuffle=False)
    args.num_samples = len(dataset_train)
    print("Number of samples:", args.num_samples)

    act = get_activations_from_dataloader(model, data_loader, args)
    #np.save("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000.npy", act)
    #np.save("./results/fid/pred_arr_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", act)
    #calculate_mmd(args, act)
    m, s = post_process(act)
    # np.save("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_m1.npy", m)
    # np.save("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_s1.npy", s)
    # np.load("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_m1ssssssss.npy")

    m1 = np.load("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_m1.npy")
    s1 = np.load("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_s1.npy")

    fid_value = calculate_frechet_distance(m1, s1, m, s)
    print('FID real: ', fid_value)

    all_imgs = []
    for batch in data_loader:
        # batch["image"]가 (B, 1, 182, 218, 182)
        all_imgs.append(batch["image"])
        if sum(x.size(0) for x in all_imgs) >= 100:
            break

    # 하나로 합치기
    gen = torch.cat(all_imgs, dim=0)

    # 1) 모든 쌍(약 N(N-1)/2) 계산 (비용 큼)
    # div, stats = ms_ssim_pairwise_diversity(gen)
    # print("Diversity =", div, "| mean MS-SSIM =", stats["mean_ms_ssim"], "| pairs =", stats["n_pairs"])

    # 2) 무작위 2,000쌍만 샘플링해서 빠르게 근사
    #div2, stats2 = ms_ssim_pairwise_diversity(gen, max_pairs=2000, seed=0)
    #print("Real Fast Diversity =", div2, "| pairs =", stats2["n_pairs"])
    #np.save("./results/fid/m_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", m)
    #np.save("./results/fid/s_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", s)
    #np.save("./results/fid/m_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", m)
    #np.save("./results/fid/s_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", s)
    return act

# def calculate_mmd_fake(args):
#     assert os.path.exists("./results/fid/pred_arr_real_"+args.real_suffix+str(args.fold)+".npy")
#     act = generate_samples(args)
#     calculate_mmd(args, act)

def calculate_fid_fake(args):
    #assert os.path.exists("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")
    args.data_dir = "/leelabsg/data/ex_MAISI/E0_UKB/predictions" # gen
    args.data_dir = "/leelabsg/data/ex_MAISI/E0_UKB/recon/whole" # recon
    args.batch_size = 2

    model = get_feature_extractor()

    train_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in os.listdir(args.data_dir) if "nii.gz" in image_name][:1000]
    datasets = {
        1: {
            "data_name": "T1_brain_to_MNI",
            "train_files": train_files,
            "modality": "mri",
        }
    }
    train_files = {"mri": []}
    # Function to add assigned class to datalist
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    # Process datasets
    for _, dataset in datasets.items():
        train_files_i = dataset["train_files"]
        print(f"{dataset['data_name']}: number of training data is {len(train_files_i)}.")
        # attach modality to each file
        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
    
    for modality in train_files.keys():
        print(f"Total number of training data for {modality} is {len(train_files[modality])}.")

    train_files_combined = train_files["mri"]
    train_transform = VAE_Transform(
        is_train=False,
        random_aug=False,  # whether apply random data augmentation for training
        k=4,  # patches should be divisible by k
        patch_size=[182, 218, 182],
        val_patch_size=None,
        output_dtype=torch.float32,  # final data type
        spacing_type="original",
        spacing=None,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    train_transform = Compose([
        train_transform,
        Resized(
            keys=["image"],
            spatial_size=(182,218,182),     # 원하는 해상도 (tuple 또는 list)
            size_mode="all",
            allow_missing_keys=True
        )
    ])
    dataset_train = CacheDataset(data=train_files_combined, transform=train_transform, cache_rate=0.0, num_workers=8)
    data_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, shuffle=False)
    args.num_samples = len(dataset_train)
    print("Number of samples:", args.num_samples)

    act = get_activations_from_dataloader(model, data_loader, args)
    m2, s2 = post_process(act)

    m1 = np.load("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_m1.npy")
    s1 = np.load("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000_s1.npy")

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('FID fake: ', fid_value)

    all_imgs = []
    for batch in data_loader:
        # batch["image"]가 (B, 1, 182, 218, 182)
        all_imgs.append(batch["image"])
        if sum(x.size(0) for x in all_imgs) >= 100:
            break

    # 하나로 합치기
    gen = torch.cat(all_imgs, dim=0)

    # 1) 모든 쌍(약 N(N-1)/2) 계산 (비용 큼)
    # div, stats = ms_ssim_pairwise_diversity(gen)
    # print("Diversity =", div, "| mean MS-SSIM =", stats["mean_ms_ssim"], "| pairs =", stats["n_pairs"])

    # 2) 무작위 2,000쌍만 샘플링해서 빠르게 근사
    div2, stats2 = ms_ssim_pairwise_diversity(gen, max_pairs=2000, seed=0)
    print("Fake Fast Diversity =", div2, "| pairs =", stats2["n_pairs"])
    return act


import torch
from itertools import combinations
from typing import Optional, Sequence
from pytorch_msssim import ms_ssim

@torch.no_grad()
def ms_ssim_pairwise_diversity(
    vols: torch.Tensor,
    max_pairs: Optional[int] = None,
    sigma: float = 1.5,
    data_range: float = 1.0,
    device: Optional[torch.device] = None,
    seed: Optional[int] = 0,
    return_matrix: bool = False,
    batch_size: int = 8,
):
    """
    vols: (N, C=1, H, W, D) 또는 (N, C=1, H, W)  ─ 값 범위는 [0,1] 권장
    max_pairs: None이면 모든 쌍(삼각행렬) 계산, 정수면 그 개수만 무작위 추출
    sigma: MS-SSIM의 가우시안 윈도우 표준편차(내부 기본값은 라이브러리 디폴트)
    data_range: 입력 값 범위 (예: [0,1] → 1.0, [-1,1] → 2.0)
    device: 계산 장치 (None이면 vols.device)
    return_matrix: True면 N×N MS-SSIM 대칭 행렬도 반환(메모리 사용↑)
    batch_size: 한 번에 평가할 (pair) 묶음 수 (GPU 메모리 조절용)

    반환:
      diversity: 1 - mean_offdiag_ms_ssim (값↑일수록 다양성↑)
      stats: dict(mean_ms_ssim, std_ms_ssim, n_pairs, per_pair=None or tensor)
      (option) ms_mat: (N,N) 대칭 행렬 (대각=1.0)
    """
    assert vols.ndim in (4, 5) and vols.size(1) == 1, "Input must be (N,1,H,W[,D])"
    N = vols.size(0)
    device = device or vols.device
    vols = vols.to(device, dtype=torch.float32)

    # 쌍 선택
    all_pairs = list(combinations(range(N), 2))
    if max_pairs is not None and max_pairs < len(all_pairs):
        g = torch.Generator(device='cpu')
        if seed is not None:
            g.manual_seed(seed)
        idx = torch.randperm(len(all_pairs), generator=g)[:max_pairs].tolist()
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    # 배치로 쌍 묶어서 MS-SSIM 계산
    ms_values = []
    C = vols.size(1)
    is_3d = (vols.ndim == 5)  # (N,1,H,W,D)

    def _ms_batch(a_idx: Sequence[int], b_idx: Sequence[int]) -> torch.Tensor:
        A = vols[torch.tensor(a_idx, device=device)]  # (B,1,*,*,[*])
        B = vols[torch.tensor(b_idx, device=device)]
        # pytorch_msssim.ms_ssim은 입력 shape: (B,C,H,W) 또는 (B,C,D,H,W)
        #val = ms_ssim(A, B, data_range=data_range, size_average=False)  # (B,)
        return _adaptive_ms_ssim(A, B, data_range=data_range, base_win=11)

    # 대칭 행렬 원하면 미리 텐서 준비
    ms_mat = None
    if return_matrix:
        ms_mat = torch.ones((N, N), device='cpu', dtype=torch.float32)

    # 배치 루프
    for start in range(0, len(pairs), batch_size):
        print(f"{start}/{len(pairs)} processing", flush=True)
        chunk = pairs[start:start+batch_size]
        a_idx = [i for i, _ in chunk]
        b_idx = [j for _, j in chunk]
        vals = _ms_batch(a_idx, b_idx).detach().float().cpu()  # (B,)
        ms_values.append(vals)
        if return_matrix:
            for (i, j), v in zip(chunk, vals):
                ms_mat[i, j] = v
                ms_mat[j, i] = v

    if len(ms_values) == 0:
        # 쌍이 1개도 없을 때 (N<2) 처리
        return 0.0, {"mean_ms_ssim": float('nan'), "std_ms_ssim": float('nan'), "n_pairs": 0, "per_pair": None}, ms_mat

    ms_values = torch.cat(ms_values, dim=0)  # (num_pairs,)
    mean_ms = ms_values.mean().item()
    std_ms = ms_values.std(unbiased=False).item()
    diversity = 1.0 - mean_ms  # 다양성 지표

    stats = {
        "mean_ms_ssim": mean_ms,
        "std_ms_ssim": std_ms,
        "n_pairs": ms_values.numel(),
        "per_pair": ms_values  # 필요 없으면 빼도 됨
    }
    return (diversity, stats, ms_mat) if return_matrix else (diversity, stats)

from math import floor, log2
from pytorch_msssim import ms_ssim, ssim

def _min_side_nd(x):
    # x: (B,1,H,W) or (B,1,D,H,W)
    return min(x.shape[-3:]) if x.ndim == 5 else min(x.shape[-2:])

def _adaptive_ms_ssim(A, B, data_range=1.0, base_win=11, min_win=3):
    """
    A,B: (B,1,H,W) or (B,1,D,H,W)
    - 입력의 최소 변 길이에 맞춰 레벨(스케일)과 win_size를 자동으로 줄임
    - 그래도 안 되면 단일 스케일 SSIM으로 폴백
    """
    base_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 최대 5레벨
    min_side = int(min(_min_side_nd(A), _min_side_nd(B)))

    # 현재 입력에서 허용되는 최대 홀수 윈도우(<= min_side)
    win = min(base_win, max(min_side, 3))
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    def levels_allowed(win_sz: int) -> int:
        """ min_side > (win-1)*2**(L-1) 를 만족하는 최대 L (1 이상) """
        if min_side <= win_sz - 1:
            return 1
        ratio = (min_side - 1) / (win_sz - 1)
        Lmax = 1 + floor(log2(ratio))  # ‘>’ 보장 위해 -1/‘floor’ 사용
        return max(1, min(Lmax, len(base_weights)))

    # 우선 현재 win에서 가능한 L 계산
    L = levels_allowed(win)

    # 가능한 한 L을 키우고 싶으면 win을 줄여보되, 항상 조건을 만족하는 범위에서만
    while L < 1 and win > min_win:
        win -= 2
        L = levels_allowed(win)
    L = max(1, L)

    if L >= 2:
        w = base_weights[:L]
        s = sum(w); w = [x / s for x in w]
        try:
            return ms_ssim(A, B, data_range=data_range, size_average=False, win_size=win, weights=w)
        except AssertionError:
            pass  # 안전 폴백
    # L==1 이거나 ms_ssim이 여전히 assert면 단일 스케일 SSIM으로 폴백
    # (win은 min_side 이하의 홀수로 보정)
    safe_win = min(win, min_side if min_side % 2 == 1 else min_side - 1)
    safe_win = max(safe_win, 3)
    return ssim(A, B, data_range=data_range, size_average=False, win_size=safe_win)
# def calculate_mmd(args, act):
#     from torch_two_sample.statistics_diff import MMDStatistic

#     act_real = np.load("./results/fid/pred_arr_real_"+args.real_suffix+str(args.fold)+".npz")['arr_0']
#     mmd = MMDStatistic(act_real.shape[0], act.shape[0])
#     sample_1 = torch.from_numpy(act_real)
#     sample_2 = torch.from_numpy(act)

#     # Need to install updated MMD package at https://github.com/lisun-ai/torch-two-sample for support of median alphas
#     test_statistics, ret_matrix = mmd(sample_1, sample_2, alphas='median', ret_matrix=True)
#     #p = mmd.pval(ret_matrix.float(), n_permutations=1000)

#     print("\nMMD test statistics:", test_statistics.item())

from sklearn.neighbors import NearestNeighbors

def calculate_precision_recall(real_features, gen_features, k=3):
    """
    K-Nearest Neighbors(k-NN)을 사용하여 Precision과 Recall을 계산합니다.
    
    Args:
        real_features (np.array): 실제 이미지에서 추출한 특징 벡터들
        gen_features (np.array): 생성 이미지에서 추출한 특징 벡터들
        k (int): k-NN 계산에 사용할 이웃의 수
    """
    num_real = real_features.shape[0]
    num_gen = gen_features.shape[0]

    # 1. 각 실제 이미지가 K개의 다른 실제 이미지와 얼마나 가까운지 계산 (다양성 manifold 추정)
    print(f"Calculating k-NN distances for {num_real} real features...")
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(real_features)
    # 각 real feature에 대해 k+1개의 최근접 이웃까지의 거리를 계산 (자기 자신 포함)
    real_distances, _ = nn_real.kneighbors(real_features)
    # 자기 자신(거리 0)을 제외하고 k번째 이웃까지의 거리를 'manifold 반지름'으로 사용
    real_radii = real_distances[:, k]

    # 2. 각 생성 이미지가 실제 이미지 manifold 안에 포함되는지 확인 (Precision)
    print(f"Calculating precision for {num_gen} generated features...")
    # 각 gen_feature에 대해 가장 가까운 real_feature 1개를 찾음
    nn_gen_to_real = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_features)
    dist_gen_to_real, indices_gen_to_real = nn_gen_to_real.kneighbors(gen_features)
    
    # 가장 가까운 real_feature의 manifold 반지름보다 가까우면 'in'
    precision_bools = dist_gen_to_real.flatten() <= real_radii[indices_gen_to_real.flatten()]
    precision = np.mean(precision_bools)

    # 3. 각 실제 이미지가 생성 이미지 manifold 안에 포함되는지 확인 (Recall)
    # (위 과정과 역할을 바꾸어 동일하게 진행)
    print(f"Calculating k-NN distances for {num_gen} generated features...")
    nn_gen = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(gen_features)
    gen_distances, _ = nn_gen.kneighbors(gen_features)
    gen_radii = gen_distances[:, k]

    print(f"Calculating recall for {num_real} real features...")
    nn_real_to_gen = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gen_features)
    dist_real_to_gen, indices_real_to_gen = nn_real_to_gen.kneighbors(real_features)
    
    recall_bools = dist_real_to_gen.flatten() <= gen_radii[indices_real_to_gen.flatten()]
    recall = np.mean(recall_bools)

    return precision, recall


if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()
    act_real = calculate_fid_real(args)
    act_fake = calculate_fid_fake(args)
    #act_fake = np.load("/shared/s1/lab06/wonyoung/maisi/eval/ukb_first1000.npy")
    precision, recall = calculate_precision_recall(act_real, act_fake, k=3)
    print("\n--- Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("----------------")
    print("Done. Using", (time.time()-start_time)//60, "minutes.")