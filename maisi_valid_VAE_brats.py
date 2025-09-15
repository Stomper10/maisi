import os
import glob
import json
import yaml
import warnings
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

import torch
from torch.amp import autocast
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader
from monai.losses.perceptual import PerceptualLoss
from monai.transforms import Compose, Resize, Resized, Compose, EnsureChannelFirst, Spacing

from utils import compute_psnr
from scripts.utils import define_instance
from scripts.transforms import VAE_Transform
from scripts.utils_plot import find_label_center_loc, get_xyz_plot

warnings.filterwarnings("ignore")

print_config()

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vae_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/results/E0_maisi_vae_1427968/checkpoint-220000")
    parser.add_argument("--model_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_vae_train.json")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)
    config_train_dict = json.load(open(args.train_config_path, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["custom_config"].items():
        setattr(args, k, v)
    return args


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




def main():
    args = load_config()
    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.weight_dtype == "fp16" else torch.float32
    input_dir = args.save_dir
    output_dir = os.path.join(args.save_dir, "slices")
    os.makedirs(output_dir, exist_ok=True)

    set_determinism(seed=args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    # Data
    valid_filepaths = [f for f in sorted(glob.glob(os.path.join(args.valid_label_dir, "*", "*.nii.gz"))) if "seg" not in f]
    val_files_list = [{"image": f} for f in valid_filepaths]

    # Function to add assigned class to datalist
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    val_files = add_assigned_class_to_datalist(val_files_list, "mri")
    
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,
        output_dtype=weight_dtype,
        image_keys=["image"]
    )
    val_transform = Compose([
        val_transform,
        Resized(
            keys=["image"],
            spatial_size=(256,256,128), # emb2와 동일한 size로 세팅 (tuple 또는 list)
            mode="trilinear"
        )
    ])

    # Build dataloader
    print(f"Total number of validation data is {len(val_files)}.")
    dataset_val = CacheDataset(data=val_files, transform=val_transform, cache_rate=0.0, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=4, shuffle=False)

    # Load pretrained model
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    path = os.path.join(args.pretrained_vae_path, "model.pt")
    ckpt = torch.load(path, map_location=device)
    autoencoder.load_state_dict(ckpt["autoencoder"])

    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    autoencoder.eval()
    val_epoch_losses = {"lpips": 0, "psnr": 0, "ssim": 0}
    for i, val_batch in enumerate(dataloader_val):
        val_images = val_batch["image"].to(device)
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                # reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images)
                reconstruction, _, _ = autoencoder(val_images)
                
                images_clip = torch.clamp(val_images, 0.0, 1.0)
                recon_clip = torch.clamp(reconstruction, 0.0, 1.0)
                
                val_epoch_losses["lpips"] += loss_perceptual(recon_clip, images_clip).item()
                val_epoch_losses["psnr"] += compute_psnr(recon_clip.cpu().numpy(), images_clip.cpu().numpy())
                val_epoch_losses["ssim"] += ssim(recon_clip.squeeze().cpu().numpy(), 
                                                 images_clip.squeeze().cpu().numpy(), 
                                                 data_range=1.0)
                
                data = reconstruction.squeeze().cpu().detach().numpy().astype(np.float32)
                image = val_images.squeeze().cpu().detach().numpy().astype(np.float32)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_size = (256,256,128)
                out_spacing = (0.9375,0.9375,1.2109375)
                output_path = "{0}/{1:03d}_recon_size{2:d}x{3:d}x{4:d}_{5}.nii.gz".format(
                    input_dir,
                    i,
                    data.shape[0],
                    data.shape[1],
                    data.shape[2],
                    timestamp,
                )
                output_path_orig = "{0}/{1:03d}_orig_size{2:d}x{3:d}x{4:d}_{5}.nii.gz".format(
                    input_dir,
                    i,
                    data.shape[0],
                    data.shape[1],
                    data.shape[2],
                    timestamp,
                )
                save_image(data, output_size, out_spacing, output_path)
                # save_image(image, output_size, out_spacing, output_path_orig)
                print(f"recon image {i:03d} saved", flush=True)

    for key in val_epoch_losses:
        val_epoch_losses[key] /= len(dataloader_val)
    print(val_epoch_losses)

    target_spacing = (1.0, 1.0, 1.0)
    target_shape = (240, 240, 155)
    final_transform = Compose([
        EnsureChannelFirst(channel_dim=0),  # (C, H, W, D)
        Spacing(
            pixdim=target_spacing,
            mode="trilinear",
            diagonal=False
        ),
        Resize(spatial_size=target_shape, mode="trilinear")
    ])

    # middle slice 추출 함수
    def load_nii_as_tensor(filepath):
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)  # → (H, W, D)
        tensor = torch.from_numpy(data).unsqueeze(0)  # → (1, H, W, D)
        return tensor
    
    def save_wandb_style_xyz_plot(tensor_1chw, filename, output_dir):
        center = find_label_center_loc(tensor_1chw[0])  # (H, W, D)
        vis_image = get_xyz_plot(tensor_1chw, center, mask_bool=False)
        vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
        vis_image = (vis_image * 255).astype(np.uint8)
        save_path = os.path.join(output_dir, f"{filename}_xyz.png")
        plt.imsave(save_path, vis_image)
        print(f"✅ Saved: {save_path}")

    # 처리 시작
    nii_paths = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    print(f"Found {len(nii_paths)} files.")

    for nii_path in nii_paths:
        tensor = load_nii_as_tensor(nii_path)  # (1, H, W, D)
        tensor = final_transform(tensor)
        base_name = os.path.basename(nii_path).replace(".nii.gz", "")
        save_wandb_style_xyz_plot(tensor, base_name, output_dir)

    print("✅ Done.")



if __name__ == '__main__':
    main()