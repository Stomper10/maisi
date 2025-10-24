import os
import time
import json
import yaml
import warnings
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import linalg
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torch.amp import autocast
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader
from monai.losses.perceptual import PerceptualLoss
from monai.transforms import (
    Compose, 
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    EnsureTyped,
    ScaleIntensityRangePercentilesd,
    Resize, 
    Resized, 
    EnsureChannelFirst,
)

from resnet3D import resnet50
from utils import compute_psnr
from scripts.utils import define_instance
from scripts.utils_plot import find_label_center_loc, get_xyz_plot

warnings.filterwarnings("ignore")

print_config()

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vae_path", type=str)
    parser.add_argument("--pretrained_unet_path", type=str)
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--vae_config_path", type=str)
    parser.add_argument("--diff_config_path", type=str)
    parser.add_argument("--eval_mode", type=str, default="real_vs_recon", 
                        choices=["real_vs_recon", "real_vs_real", "real_vs_gen"],
                        help="Evaluation mode to run.")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_label_dir", type=str)
    parser.add_argument("--other_label_dir", type=str)
    args = parser.parse_args()
    model_config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in model_config_dict.items():
        setattr(args, k, v)
    vae_config_dict = json.load(open(args.vae_config_path, "r"))
    for k, v in vae_config_dict["data_option"].items():
        setattr(args, k, v)
    for k, v in vae_config_dict["autoencoder_train"].items():
        setattr(args, k, v)
    for k, v in vae_config_dict["custom_config"].items():
        setattr(args, k, v)
    diff_config_dict = json.load(open(args.diff_config_path, "r"))
    for k, v in diff_config_dict.items():
        setattr(args, k, v)
    return args

def _pairwise_distances(x, y):
    # x: [N, D], y: [M, D]
    # returns [N, M]
    x2 = np.sum(x**2, axis=1, keepdims=True)  # (N,1)
    y2 = np.sum(y**2, axis=1, keepdims=True).T  # (1,M)
    xy = x @ y.T
    # numerical stability
    d2 = np.maximum(x2 + y2 - 2 * xy, 0.0)
    return np.sqrt(d2, dtype=np.float64)

def compute_prdc(real_features, gen_features, k=3):
    """
    real_features: (N_r, D), gen_features: (N_g, D), both float64/float32
    Implements Kynkäänniemi+ (Improved Precision/Recall) and Naeem+ (Density/Coverage).
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    gen_features  = np.asarray(gen_features,  dtype=np.float64)

    # kNN radii
    d_rr = _pairwise_distances(real_features, real_features)
    # exclude self (set diagonal to +inf) to mimic "k-th neighbor other than self"
    np.fill_diagonal(d_rr, np.inf)
    r_real = np.partition(d_rr, kth=k-1, axis=1)[:, k-1]  # radius for each real

    d_gg = _pairwise_distances(gen_features, gen_features)
    np.fill_diagonal(d_gg, np.inf)
    r_gen = np.partition(d_gg, kth=k-1, axis=1)[:, k-1]   # radius for each gen

    # cross distances
    d_gr = _pairwise_distances(gen_features, real_features)  # (N_g, N_r)
    d_rg = d_gr.T                                           # (N_r, N_g)

    # Precision: gen within some real ball
    nearest_real_idx_for_gen = np.argmin(d_gr, axis=1)         # (N_g,)
    prec = np.mean(d_gr[np.arange(d_gr.shape[0]), nearest_real_idx_for_gen]
                   <= r_real[nearest_real_idx_for_gen])

    # Recall: real within some gen ball (uses gen radii)
    nearest_gen_idx_for_real = np.argmin(d_rg, axis=1)          # (N_r,)
    rec = np.mean(d_rg[np.arange(d_rg.shape[0]), nearest_gen_idx_for_real]
                  <= r_gen[nearest_gen_idx_for_real])

    # Coverage: fraction of real covered by any gen within r_real(real)
    min_dist_real_to_any_gen = np.min(d_rg, axis=1)             # (N_r,)
    cov = np.mean(min_dist_real_to_any_gen <= r_real)

    # Density: for each gen, how many real balls contain it, divided by k, averaged across gen
    # Indicator matrix: real j includes gen i if d_gr[i, j] <= r_real[j]
    includes = (d_gr <= r_real[None, :])                        # (N_g, N_r)
    den = np.mean(np.sum(includes, axis=1) / float(k))          # average over gen
    return float(prec), float(rec), float(den), float(cov)

def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

def trim_state_dict_name(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

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

def get_activations_from_dataloader(model, data_loader, dataset_len):
    pred_arr = np.empty((dataset_len, 2048)) # args.dims -> 2048
    start_idx = 0
    for i, batch in enumerate(data_loader):
        if i % 10 == 0:
            print(f'\rPropagating batch {i}/{len(data_loader)}', end='', flush=True)
        
        batch_images = batch[0].float().cuda()
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

def calculate_precision_recall(real_features, gen_features, k=3): # l2 euclidean
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

def load_nii_as_tensor(filepath):
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)  # → (H, W, D)
    tensor = torch.from_numpy(data).unsqueeze(0)  # → (1, H, W, D)
    return tensor
    
def save_wandb_style_xyz_plot(tensor_1chw, filename, output_dir):
    center = find_label_center_loc(tensor_1chw[0])  # (H, W, D)
    center = [89, 110, 89]
    vis_image = get_xyz_plot(tensor_1chw, center, mask_bool=False)
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
    vis_image = (vis_image * 255).astype(np.uint8)
    save_path = os.path.join(output_dir, f"{filename}_xyz.png")
    plt.imsave(save_path, vis_image)
    print(f"✅ Saved: {save_path}")

def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data (np.ndarray): Synthetic image data.
        output_size (tuple): Output size of the image.
        out_spacing (tuple): Spacing of the output image.
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)

def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

def main():
    start_time = time.time()
    args = load_config()
    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.weight_dtype == "fp16" else torch.float32
    slice_dir = os.path.join(args.save_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    base_temp_dir = os.path.join(args.save_dir, "base_temp")
    other_temp_dir = os.path.join(args.save_dir, "other_temp")
    os.makedirs(base_temp_dir, exist_ok=True)
    os.makedirs(other_temp_dir, exist_ok=True)

    set_determinism(seed=args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    num_evaluate = 500
    SEEDS = [21, 42, 111, 123, 450, 555, 654, 777, 984, 1000,
             2011, 2024, 3000, 3456, 4500, 5000, 6789, 8888, 9854, 9999]
    
    transform = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
        Resized(keys=["image"], spatial_size=(128, 256, 128), mode="trilinear"),
        EnsureTyped(keys="image", dtype=weight_dtype),
    ])
    gen_transform = Compose([
        # 생성된 이미지도 동일하게 0~1 사이로 스케일링
        ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
        EnsureTyped(keys="image", dtype=weight_dtype),
    ])
    slice_transform = Compose([
        EnsureChannelFirst(channel_dim=0),  # (C, H, W, D)
        Resize(spatial_size=(182, 218, 182), mode="trilinear")
    ])

    if args.eval_mode != "real_vs_real":
        autoencoder = define_instance(args, "autoencoder_def").to(device)
        path = os.path.join(args.pretrained_vae_path, "model.pt")
        ckpt = torch.load(path, map_location=device)
        autoencoder.load_state_dict(ckpt["autoencoder"])
        autoencoder.eval()
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)    
    
    if args.eval_mode == "real_vs_gen":
        noise_scheduler = define_instance(args, "noise_scheduler")
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor([32, 64, 32])),
        )
        unet = define_instance(args, "diffusion_unet_def").to(device)
        path = os.path.join(args.pretrained_unet_path, "diff_unet_ckpt.pt")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
        unet.eval()
        scale_factor = ckpt["scale_factor"]

    final_results = {}

    base_label_df = pd.read_csv(args.base_label_dir)[:num_evaluate]
    base_files_list = [{"image": os.path.join("/leelabsg/data/20252_unzip/", image_name)} for image_name in base_label_df["rel_path"]]
    base_files = add_assigned_class_to_datalist(base_files_list, "mri")
    
    print(f"Total number of real (base) data is {len(base_files)}.")
    dataset_base = CacheDataset(data=base_files, transform=transform, cache_rate=0.0, num_workers=4)
    dataloader_base = DataLoader(dataset_base, batch_size=1, num_workers=4, shuffle=False)

    feature_extractor = get_feature_extractor()
    base_volumes, other_volumes = [], []
    # base_files_for_features = []
    # other_files_for_features = []

    # # Data
    if args.eval_mode == "real_vs_real":
        for i, base_batch in enumerate(dataloader_base):
            print(f"\rProcessing batch {i+1}/{len(dataloader_base)}...", end='', flush=True)
            base_images = base_batch["image"].to(device)
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    base_volumes.append(base_images.cpu())
                    save_wandb_style_xyz_plot(slice_transform(base_images.cpu()[0]), 
                                              f"{i:03d}_base", slice_dir)
                    # base_output_path = os.path.join(base_temp_dir, f"base_{i:04d}.nii.gz")
                    # nib.save(nib.Nifti1Image(base_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), base_output_path)
                    # base_files_for_features.append({"image": base_output_path})

        other_label_df = pd.read_csv(args.other_label_dir)[:num_evaluate]
        other_files_list = [{"image": os.path.join("/leelabsg/data/20252_unzip/", image_name)} for image_name in other_label_df["rel_path"]]
        other_files = add_assigned_class_to_datalist(other_files_list, "mri")

        # Build dataloader
        print(f"Total number of real (other) data is {len(other_files)}.")
        dataset_other = CacheDataset(data=other_files, transform=transform, cache_rate=0.0, num_workers=4)
        dataloader_other = DataLoader(dataset_other, batch_size=1, num_workers=4, shuffle=False)

        for i, other_batch in enumerate(dataloader_other):
            print(f"\rProcessing batch {i+1}/{len(dataloader_other)}...", end='', flush=True)
            other_images = other_batch["image"].to(device)
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    other_volumes.append(other_images.cpu())
                    save_wandb_style_xyz_plot(slice_transform(other_images.cpu()[0]), 
                                              f"{i:03d}_other", slice_dir)
                    # other_output_path = os.path.join(other_temp_dir, f"other_{i:04d}.nii.gz")
                    # nib.save(nib.Nifti1Image(other_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), other_output_path)
                    # other_files_for_features.append({"image": other_output_path})

    if args.eval_mode == "real_vs_recon":
        recon_losses = {"lpips": 0, "psnr": 0 , "ssim": 0}
        for i, base_batch in enumerate(dataloader_base):
            print(f"\rProcessing batch {i+1}/{len(dataloader_base)}...", end='', flush=True)
            base_images = base_batch["image"].to(device)
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    base_volumes.append(base_images.cpu())
                    save_wandb_style_xyz_plot(slice_transform(base_images.cpu()[0]), 
                                              f"{i:03d}_base", slice_dir)
                    
                    # reconstruction, _, _ = autoencoder(base_images)
                    # other_volumes.append(reconstruction.cpu())
                    # save_wandb_style_xyz_plot(slice_transform(reconstruction.cpu()[0]), 
                    #                           f"{i:03d}_recon", slice_dir)

                    # images_clip = torch.clamp(base_images, 0.0, 1.0)
                    # recon_clip = torch.clamp(reconstruction, 0.0, 1.0)
                
                    # recon_losses["lpips"] += loss_perceptual(recon_clip, images_clip).item()
                    # recon_losses["psnr"] += compute_psnr(recon_clip.cpu().numpy(), images_clip.cpu().numpy())
                    # recon_losses["ssim"] += ssim(recon_clip.squeeze().cpu().numpy(), 
                    #                              images_clip.squeeze().cpu().numpy(), 
                    #                              data_range=1.0)
                
                    # base_output_path = os.path.join(base_temp_dir, f"base_{i:04d}.nii.gz")
                    # nib.save(nib.Nifti1Image(base_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), base_output_path)
                    # base_files_for_features.append({"image": base_output_path})

                    # other_output_path = os.path.join(other_temp_dir, f"recon_{i:04d}.nii.gz")
                    # nib.save(nib.Nifti1Image(reconstruction.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), other_output_path)
                    # other_files_for_features.append({"image": other_output_path})
                    
        # for key in recon_losses: recon_losses[key] /= len(dataloader_base)
        # print("\nReconstruction Metrics:", recon_losses)

        other_label_df = pd.read_csv(args.other_label_dir)[:num_evaluate]
        other_files_list = [{"image": os.path.join("/leelabsg/data/20252_unzip/", image_name)} for image_name in other_label_df["rel_path"]]
        other_files = add_assigned_class_to_datalist(other_files_list, "mri")
        print(f"Total number of recon (other) data is {len(other_files)}.")
        dataset_other = CacheDataset(data=other_files, transform=transform, cache_rate=0.0, num_workers=4)
        dataloader_other = DataLoader(dataset_other, batch_size=1, num_workers=4, shuffle=False)

        for i, other_batch in enumerate(dataloader_other):
            print(f"\rProcessing batch {i+1}/{len(dataloader_other)}...", end='', flush=True)
            other_images = other_batch["image"].to(device)
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    reconstruction, _, _ = autoencoder(other_images)
                    other_volumes.append(reconstruction.cpu())
                    save_wandb_style_xyz_plot(slice_transform(reconstruction.cpu()[0]), 
                                              f"{i:03d}_recon_other", slice_dir)

    if args.eval_mode == "real_vs_gen":
        for i, base_batch in enumerate(dataloader_base):
            print(f"\rProcessing batch {i+1}/{len(dataloader_base)}...", end='')
            base_images = base_batch["image"].to(device)
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    base_volumes.append(base_images.cpu())
                    save_wandb_style_xyz_plot(slice_transform(base_images.cpu()[0]), 
                                              f"{i:03d}_base", slice_dir)
                    # base_output_path = os.path.join(base_temp_dir, f"base_{i:04d}.nii.gz")
                    # nib.save(nib.Nifti1Image(base_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), base_output_path)
                    # base_files_for_features.append({"image": base_output_path})

        # 제공된 통계치를 기반으로 각 구간 정의
        stats = {
            "min": 0.0, "p05": 0.1667, "p25": 0.3333,
            "p75": 0.6667, "p95": 0.8333, "max": 1.0
        }

        # 각 구간에서 균등하게 샘플링
        s1 = np.random.uniform(stats["min"], stats["p05"], int(num_evaluate * 0.05))
        s2 = np.random.uniform(stats["p05"], stats["p25"], int(num_evaluate * 0.20))
        s3 = np.random.uniform(stats["p25"], stats["p75"], int(num_evaluate * 0.50))
        s4 = np.random.uniform(stats["p75"], stats["p95"], int(num_evaluate * 0.20))
        s5 = np.random.uniform(stats["p95"], stats["max"], int(num_evaluate * 0.05))

        # 모든 샘플을 하나로 합치고 섞어줌
        meta_values = np.concatenate([s1, s2, s3, s4, s5])
        np.random.shuffle(meta_values)
        for i in trange(len(dataloader_base), desc=f"Generating samples"):
            print(f"\rProcessing batch {i+1}/{len(dataloader_base)}...", end='')
            torch.manual_seed(i)
            noise = torch.randn((1,4,32,64,32), device=device)
            latent = noise
            spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2
            spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
            meta_value = meta_values[i]
            meta_tensor = torch.tensor([[meta_value]], device=device, dtype=torch.float16) ### 👈
            all_timesteps = noise_scheduler.timesteps
            all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    for t, next_t in zip(all_timesteps, all_next_timesteps):
                        # Create a dictionary to store the inputs
                        unet_inputs = {
                            "x": latent,
                            "timesteps": torch.Tensor((t,)).to(device),
                            "spacing_tensor": spacing_tensor,
                            "meta_tensor": meta_tensor ### 👈
                        }
                        model_output = unet(**unet_inputs)
                        latent, _ = noise_scheduler.step(model_output, t, latent, next_t, args.diffusion_unet_inference["stochastic_scale"]) 
                    synthetic_images = autoencoder.decode_stage_2_outputs(latent / scale_factor)
                    synthetic_dict = {"image": synthetic_images.cpu()} 
                    transformed_synthetic = gen_transform(synthetic_dict)["image"]
                    other_volumes.append(transformed_synthetic)
                    save_wandb_style_xyz_plot(slice_transform(synthetic_images.cpu()[0]), 
                                              f"{i:03d}_gen", slice_dir)
                    # other_output_path = os.path.join(other_temp_dir, f"gen_{i:04d}.nii.gz")
                    # nib.save(nib.Nifti1Image(synthetic_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), other_output_path)
                    # other_files_for_features.append({"image": other_output_path})
                    
    print(f"\n--- Starting Distribution Evaluation: {args.eval_mode} ---")
    # 리스트의 텐서들을 하나로 합침
    all_base_volumes = torch.cat(base_volumes)
    all_other_volumes = torch.cat(other_volumes)

    # 메모리에 올라온 텐서들로 새로운 데이터로더 생성
    dataset_base_mem = torch.utils.data.TensorDataset(all_base_volumes)
    dataloader_base_mem = torch.utils.data.DataLoader(dataset_base_mem, batch_size=16, num_workers=4)
    
    dataset_other_mem = torch.utils.data.TensorDataset(all_other_volumes)
    dataloader_other_mem = torch.utils.data.DataLoader(dataset_other_mem, batch_size=16, num_workers=4)

    # base_files_for_features = [{"image": os.path.join(base_temp_dir, nii_file)} for nii_file in os.listdir("/leelabsg/data/ex_MAISI/maisi_ukb/base_temp")]
    # other_files_for_features = [{"image": os.path.join(other_temp_dir, nii_file)} for nii_file in os.listdir("/leelabsg/data/ex_MAISI/maisi_ukb/other_temp")]

    # feature_extraction_transform = transform
    # # Base 데이터로더
    # dataset_base_features = CacheDataset(data=base_files_for_features, transform=feature_extraction_transform, cache_rate=0.0, num_workers=4)
    # dataloader_base_features = DataLoader(dataset_base_features, batch_size=16, num_workers=4)
    
    # # Other 데이터로더
    # dataset_other_features = CacheDataset(data=other_files_for_features, transform=feature_extraction_transform, cache_rate=0.0, num_workers=4)
    # dataloader_other_features = DataLoader(dataset_other_features, batch_size=16, num_workers=4)

    # 특징 추출
    # print("Extracting features from base volumes...")
    # act_base_full = get_activations_from_dataloader(feature_extractor, dataloader_base_features, len(dataset_base_features))
    # print(f"Extracting features from {args.eval_mode.split('_vs_')[1]} volumes...")
    # act_other_full = get_activations_from_dataloader(feature_extractor, dataloader_other_features, len(dataset_other_features))

    print("Extracting features from base volumes...")
    act_base_full = get_activations_from_dataloader(feature_extractor, dataloader_base_mem, len(dataset_base_mem))
    print(f"Extracting features from {args.eval_mode.split('_vs_')[1]} volumes...")
    act_other_full = get_activations_from_dataloader(feature_extractor, dataloader_other_mem, len(dataset_other_mem))

    # --- 2. 부트스트래핑 실행 ---
    print(f"\n--- Starting bootstrapping process ---")
    num_available = min(len(act_base_full), len(act_other_full))
    num_samples_to_use = int(num_available * 0.9)
    print(f"Number of samples for each bootstrap iteration: {num_samples_to_use} (90% of {num_available})")

    fid_scores, precision_scores, recall_scores, density_scores, coverage_scores = [], [], [], [], []
    
    for i, seed in enumerate(SEEDS):
        np.random.seed(seed)
        base_indices = np.random.choice(len(act_base_full), num_samples_to_use, replace=False)
        other_indices = np.random.choice(len(act_other_full), num_samples_to_use, replace=False)
        
        act_base_subset = act_base_full[base_indices]
        act_other_subset = act_other_full[other_indices]

        # 모든 메트릭 계산
        m_base, s_base = post_process(act_base_subset)
        m_other, s_other = post_process(act_other_subset)
        fid_scores.append(calculate_frechet_distance(m_base, s_base, m_other, s_other))
        # precision, recall = calculate_precision_recall(act_base_subset, act_other_subset, k=3)
        # precision_scores.append(precision)
        # recall_scores.append(recall)
        # density, coverage = calculate_density_coverage(act_base_subset, act_other_subset, k=3)
        # density_scores.append(density)
        # coverage_scores.append(coverage)
        prec, rec, den, cov = compute_prdc(act_base_subset, act_other_subset, k=3)
        precision_scores.append(prec)
        recall_scores.append(rec)
        density_scores.append(den)
        coverage_scores.append(cov)
        
        print(f'\rIteration {i+1}/{len(SEEDS)} done.', end='', flush=True)
    print("\nBootstrapping finished.")

    # --- 3. 결과 저장 및 출력 ---
    final_results = {
        "FID": (np.mean(fid_scores), np.std(fid_scores)),
        "Precision": (np.mean(precision_scores), np.std(precision_scores)),
        "Recall": (np.mean(recall_scores), np.std(recall_scores)),
        "Density": (np.mean(density_scores), np.std(density_scores)),
        "Coverage": (np.mean(coverage_scores), np.std(coverage_scores)),
    }

    print(f"\n--- Final Results (Mean ± Std. Dev. over {len(SEEDS)} runs) ---")
    print(f"FID:                 {final_results['FID'][0]:.4f} ± {final_results['FID'][1]:.4f}")
    print(f"Precision:           {final_results['Precision'][0]:.4f} ± {final_results['Precision'][1]:.4f}")
    print(f"Recall:              {final_results['Recall'][0]:.4f} ± {final_results['Recall'][1]:.4f}")
    print(f"Density:             {final_results['Density'][0]:.4f} ± {final_results['Density'][1]:.4f}")
    print(f"Coverage:            {final_results['Coverage'][0]:.4f} ± {final_results['Coverage'][1]:.4f}")
    print(f"Total time: {(time.time()-start_time)/60:.2f} minutes.")
    print("✅ Done.")



if __name__ == '__main__':
    main()