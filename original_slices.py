import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scripts.utils_plot import find_label_center_loc, get_xyz_plot
import matplotlib.pyplot as plt
from monai.transforms import Resize, Compose, EnsureChannelFirst, Spacing, ScaleIntensityRangePercentiles

# Data
valid_label_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv"
data_dir = "/leelabsg/data/20252_unzip/"
valid_label_df = pd.read_csv(valid_label_dir)[1000:1100]
valid_files = [os.path.join(data_dir, image_name) for image_name in valid_label_df["rel_path"]]
output_dir = "/leelabsg/data/ex_MAISI/E0_UKB/original/whole_trans"
target_spacing = (1.0, 1.0, 1.0)
target_shape = (182, 218, 182)
transform = Compose([
        EnsureChannelFirst(channel_dim=0),  # (C, H, W, D)
        Spacing(
            pixdim=target_spacing,
            mode="trilinear",
            diagonal=False
        ),
        ScaleIntensityRangePercentiles(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
        Resize(spatial_size=target_shape, mode="trilinear")
    ])

# middle slice 추출 함수
def load_nii_as_tensor(filepath):
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)  # → (H, W, D)
    tensor = torch.from_numpy(data).unsqueeze(0)  # → (1, H, W, D)
    return tensor

def save_wandb_style_xyz_plot(tensor_1chw, filename, output_dir):
    #center = find_label_center_loc(tensor_1chw[0])  # (H, W, D)
    center = torch.tensor([89,110,89])
    vis_image = get_xyz_plot(tensor_1chw, center, mask_bool=False)
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
    vis_image = (vis_image * 255).astype(np.uint8)
    # ✅ 1st: percentile-based intensity clipping (e.g., 1% ~ 99%)
    # vmin, vmax = np.percentile(vis_image, [1, 99])
    # vis_image = np.clip(vis_image, vmin, vmax)

    # # ✅ 2nd: normalize to 0-255 for display
    # vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
    # vis_image = (vis_image * 255).astype(np.uint8)

    save_path = os.path.join(output_dir, f"{filename}_xyz.png")
    plt.imsave(save_path, vis_image)
    print(f"✅ Saved: {save_path}")

# 처리 시작
for i, nii_path in enumerate(valid_files):
    # Load and preprocess
    tensor = load_nii_as_tensor(nii_path)  # (1, H, W, D)
    tensor = transform(tensor)
    base_name = os.path.basename(nii_path).replace(".nii.gz", f"_{i}")
    save_wandb_style_xyz_plot(tensor, base_name, output_dir)

print("✅ Done.")