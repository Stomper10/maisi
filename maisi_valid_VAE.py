import os
import time
import json
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.nn.functional import adaptive_avg_pool3d
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.perceptual import PerceptualLoss
from monai.utils import set_determinism
from torch.amp import autocast
from monai.transforms import Compose, Resized

from utils import count_parameters, compute_psnr, compute_fid, clip_and_normalize, PretrainedMedicalModel
from scripts.transforms import VAE_Transform
from scripts.utils import define_instance, dynamic_infer

#from torchmetrics.functional import peak_signal_noise_ratio as psnr
#from torchmetrics.functional import structural_similarity_index_measure as ssim
from skimage.metrics import structural_similarity as ssim
#from torchmetrics.image.fid import FrechetInceptionDistance
from scripts.utils_plot import find_label_center_loc, get_xyz_plot #, show_image
from PIL import Image
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

print_config()

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vae_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/results/E0_maisi_vae_1427968/checkpoint-220000")
    parser.add_argument("--model_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_vae_train.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)
    config_train_dict = json.load(open(args.train_config_path, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
        #print(f"{k}: {v}")
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
        #print(f"{k}: {v}")
    for k, v in config_train_dict["custom_config"].items():
        setattr(args, k, v)
    return args

def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    #logger: logging.Logger,
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
    #logger.info(f"Saved {output_path}.")



def main():
    args = load_config()
    device = torch.device(args.device)

    weight_dtype = torch.float32
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    set_determinism(seed=args.seed)

    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    # Data
    valid_label_df = pd.read_csv(args.valid_label_dir)[1000:2000]
    valid_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in valid_label_df["rel_path"]]

    # Expandable to more datasets
    datasets = {
        1: {
            "data_name": "T1_brain_to_MNI",
            "val_files": valid_files,
            "modality": "mri",
        }
    }

    # Build training dataset
    # Initialize file lists
    val_files = {"mri": []}

    # Function to add assigned class to datalist
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    # Process datasets
    for _, dataset in datasets.items():
        val_files_i = dataset["val_files"]
        print(f"{dataset['data_name']}: number of val data is {len(val_files_i)}.")

        # attach modality to each file
        modality = dataset["modality"]
        val_files[modality] += add_assigned_class_to_datalist(val_files_i, modality)

    # Print total numbers for each modality
    for modality in val_files.keys():
        print(f"Total number of val data for {modality} is {len(val_files[modality])}.")

    # Combine the data
    val_files_combined = val_files["mri"]
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        patch_size=[182, 218, 182],
        val_patch_size=None,  # if None, will validate on whole image volume
        output_dtype=torch.float32,  # final data type
        spacing_type="original",
        spacing=None,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
        #resolution=(182,218,182) ###
    )
    val_transform = Compose([
        val_transform,
        Resized(
            keys=["image"],
            spatial_size=(128,256,128),     # 원하는 해상도 (tuple 또는 list)
            size_mode="all",
            allow_missing_keys=True
        )
    ])

    # Build dataloader
    print(f"Total number of validation data is {len(val_files_combined)}.")
    dataset_val = CacheDataset(data=val_files_combined, transform=val_transform, cache_rate=args.cache, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=4, shuffle=False)

    # Load pretrained model
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    path = os.path.join(args.pretrained_vae_path, "model.pt")
    ckpt = torch.load(path, map_location=device)
    autoencoder.load_state_dict(ckpt["autoencoder"])

    loss_perceptual = (
        PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    )
    # fid_model = PretrainedMedicalModel().eval().to(device)

    # def get_latent_features(volumes):
    #     with torch.no_grad():
    #         feats = fid_model(volumes.to(device))
    #         pooled = adaptive_avg_pool3d(feats[1], output_size=2)  # or 2, or 1
    #         return pooled.flatten(1).cpu().numpy()  # (B, C×4×4×4)
    
    #fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    # Training config
    param_counts = count_parameters(autoencoder)
    print(f"### autoencoder's Total parameters: {param_counts['total']:,}")
    print(f"### autoencoder's Trainable parameters: {param_counts['trainable']:,}")
    print(f"### autoencoder's Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"### autoencoder's Parameters by layer type: {param_counts['by_layer_type']}")    

    # Training
    # Setup validation inferer
    # print("val_sliding_window_patch_size: ", args.val_sliding_window_patch_size, flush=True)
    # val_inferer = (
    #     SlidingWindowInferer(
    #         roi_size=args.val_sliding_window_patch_size,
    #         sw_batch_size=1,
    #         progress=True,
    #         mode="gaussian",
    #         overlap=0.25,
    #         device=device,
    #         sw_device=device,
    #     )
    #     if args.val_sliding_window_patch_size
    #     else SimpleInferer()
    # )

    autoencoder.eval()
    val_epoch_losses = {"lpips": 0, "psnr": 0, "ssim": 0}
    # all_feats_recon, all_feats_gt = [], []
    start_time = time.time()
    for i, batch in enumerate(dataloader_val):
        with torch.no_grad():
            with autocast("cuda", enabled=args.amp):
                images = batch["image"].to(device).contiguous()
                # reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images)
                reconstruction, _, _ = autoencoder(images)
                print(images.shape, flush=True)
                print(reconstruction.shape, flush=True)
                
                images_clip = clip_and_normalize(images)
                recon_clip = clip_and_normalize(reconstruction)
                
                val_epoch_losses["lpips"] += loss_perceptual(recon_clip, images_clip).item()
                val_epoch_losses["psnr"] += compute_psnr(recon_clip.cpu().numpy(), images_clip.cpu().numpy())
                val_epoch_losses["ssim"] += ssim(recon_clip.squeeze().cpu().numpy(), 
                                                 images_clip.squeeze().cpu().numpy(), 
                                                 data_range=1.0)
                # FID
                # all_feats_recon.append(get_latent_features(recon_clip))
                # all_feats_gt.append(get_latent_features(images_clip))

                # print(val_epoch_losses, flush=True)
                data = reconstruction.squeeze().cpu().detach().numpy().astype(np.float32)
                # v_min, v_max = np.percentile(data, 0), np.percentile(data, 99.5)
                # b_min, b_max = 0.0, 1.0
                # if v_max == v_min:
                #     return np.zeros_like(data, dtype=np.float32)
                # normed = (data - v_min) / (v_max - v_min)
                # normed = normed * (b_max - b_min) + b_min

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_size = (128,256,128)
                out_spacing = (1.421875, 0.8515625, 1.421875)
                output_path = "{0}/recon_size{1:d}x{2:d}x{3:d}_{4}.nii.gz".format(
                    "/leelabsg/data/ex_MAISI/E0_UKB/recon/whole",
                    data.shape[0],
                    data.shape[1],
                    data.shape[2],
                    timestamp,
                )
                save_image(data, output_size, out_spacing, output_path)
                print(f"recon image {timestamp} saved", flush=True)

    # Monitor reconstruction result
    center_loc_axis = find_label_center_loc(images[0, 0, ...])
    vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())
    vis_image = (vis_image * 255).astype(np.uint8)
    vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)
    vis_recon_image = (vis_recon_image - vis_recon_image.min()) / (vis_recon_image.max() - vis_recon_image.min())
    vis_recon_image = (vis_recon_image * 255).astype(np.uint8)
    Image.fromarray(vis_image).save("/shared/s1/lab06/wonyoung/maisi/images/vis_image.png")
    Image.fromarray(vis_recon_image).save("/shared/s1/lab06/wonyoung/maisi/images/vis_recon_image.png")

    # # 2. 모든 배치 처리 후, 평균과 공분산 계산
    # feat_recon = np.concatenate(all_feats_recon, axis=0)
    # feat_gt = np.concatenate(all_feats_gt, axis=0)
    # mu1, sigma1 = feat_recon.mean(axis=0), np.cov(feat_recon, rowvar=False)
    # mu2, sigma2 = feat_gt.mean(axis=0), np.cov(feat_gt, rowvar=False)
    # rfid_score = compute_fid(mu1, sigma1, mu2, sigma2)

    time_elapsed = time.time() - start_time

    for key in val_epoch_losses:
        val_epoch_losses[key] /= len(dataloader_val)
    #val_epoch_losses["rfid"] = rfid_score
    print(val_epoch_losses)
    print(f"Total valid time: {time_elapsed:.2f} sec")



if __name__ == '__main__':
    main()