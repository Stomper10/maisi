import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

def load_nifti_to_tensor(filepath):
    nii = nib.load(filepath)
    image = nii.get_fdata().astype(np.float32)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    return image_tensor  # shape: [1,1,D,H,W]

def fft3d(image_tensor):
    image_complex = image_tensor.to(torch.complex64)
    kspace = torch.fft.fftn(image_complex, dim=(-3, -2, -1))
    return kspace  # shape: [1,1,D,H,W] complex

def check_symmetry(kspace):
    # flip & conj
    kspace_sym = torch.conj(torch.flip(kspace, dims=[-3, -2, -1]))
    diff = torch.abs(kspace - kspace_sym)  # element-wise error
    return diff[0, 0]  # remove batch/channel dims

def visualize_symmetry_error(diff_tensor, slice_axis=0):
    diff_np = diff_tensor.cpu().numpy()
    if slice_axis == 0:
        slice_ = diff_np[diff_np.shape[0] // 2]
    elif slice_axis == 1:
        slice_ = diff_np[:, diff_np.shape[1] // 2, :]
    else:
        slice_ = diff_np[:, :, diff_np.shape[2] // 2]

    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(slice_), cmap='hot')
    plt.title('log(abs(k - flip(conj(k))))')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("/shared/s1/lab06/wonyoung/maisi/error.png")
    plt.close()

# ✅ 실행
nifti_path = "/leelabsg/data/20252_unzip/1000502_20252_2_0/T1/T1_brain_to_MNI.nii.gz"  # 여기에 실제 경로 넣기
image = load_nifti_to_tensor(nifti_path)
kspace = fft3d(image)
diff = check_symmetry(kspace)
visualize_symmetry_error(diff, slice_axis=0)  # 0, 1, 2 중 선택 가능

def reconstruct_kspace_from_half_3d(kspace):
    """
    Use half of the k-space volume and reconstruct the rest using Hermitian symmetry.
    Assumes kspace shape: [1, 1, D, H, W], torch.complex64
    """
    assert kspace.ndim == 5 and kspace.shape[1] == 1
    B, C, D, H, W = kspace.shape

    kspace_recon = torch.zeros_like(kspace)

    # Generate full 3D grid
    z = torch.arange(D)
    y = torch.arange(H)
    x = torch.arange(W)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Mask for positive-frequency half-space (includes DC plane)
    mask = (
        (zz < D // 2)
        | ((zz == D // 2) & (yy < H // 2))
        | ((zz == D // 2) & (yy == H // 2) & (xx <= W // 2))
    )

    # Copy only half of k-space
    kspace_recon[0, 0][mask] = kspace[0, 0][mask]

    # Hermitian conjugate completion
    zz_c = (-zz) % D
    yy_c = (-yy) % H
    xx_c = (-xx) % W

    # Fill the other half from conjugate
    mask_c = ~mask
    kspace_recon[0, 0][mask_c] = torch.conj(kspace[0, 0][zz_c[mask_c], yy_c[mask_c], xx_c[mask_c]])

    return kspace_recon


def recover_image_from_kspace(kspace):
    image_recon = torch.fft.ifftn(kspace, dim=(-3,-2,-1)).real  # only real part
    return image_recon


img_orig = image[0,0]  # original real-valued image
k_recon = reconstruct_kspace_from_half_3d(kspace)
img_recon = recover_image_from_kspace(k_recon)[0,0]

# PSNR, SSIM 등 비교
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

img_orig_np = img_orig.cpu().numpy()
img_recon_np = img_recon.cpu().numpy()

print("PSNR:", psnr(img_orig_np, img_recon_np, data_range=img_orig_np.max()))
print("SSIM:", ssim(img_orig_np, img_recon_np, data_range=img_orig_np.max()))

import matplotlib.pyplot as plt

def plot_mri_comparison(img_orig, img_recon, save_path="mri_comparison.png", vmin=None, vmax=None):
    """
    Compare original and reconstructed MRI in axial, coronal, and sagittal views.
    
    Args:
        img_orig: torch.Tensor or np.ndarray of shape [D, H, W]
        img_recon: same shape as img_orig
        save_path: file path to save the figure
        vmin, vmax: value range for consistent intensity scaling
    """
    if isinstance(img_orig, torch.Tensor):
        img_orig = img_orig.cpu().numpy()
    if isinstance(img_recon, torch.Tensor):
        img_recon = img_recon.cpu().numpy()

    d, h, w = img_orig.shape
    slices = {
        "Axial (Z)":    (img_orig[d//2], img_recon[d//2]),
        "Coronal (Y)":  (img_orig[:, h//2, :], img_recon[:, h//2, :]),
        "Sagittal (X)": (img_orig[:, :, w//2], img_recon[:, :, w//2]),
    }

    if vmin is None:
        vmin = min(img_orig.min(), img_recon.min())
    if vmax is None:
        vmax = max(img_orig.max(), img_recon.max())

    plt.figure(figsize=(10, 9))
    for i, (title, (s1, s2)) in enumerate(slices.items()):
        plt.subplot(3, 2, 2*i + 1)
        plt.imshow(s1, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"{title} - Original")
        plt.axis('off')

        plt.subplot(3, 2, 2*i + 2)
        plt.imshow(s2, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"{title} - Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# 정규화 함수
def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

img_orig_norm = normalize(img_orig)
img_recon_norm = normalize(img_recon.abs())

plot_mri_comparison(
    img_orig_norm,
    img_recon_norm,
    save_path="comparison_norm.png",
    vmin=0,
    vmax=1
)


#######################################

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_nifti_to_tensor(filepath):
    nii = nib.load(filepath)
    image = nii.get_fdata().astype(np.float32)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    return image_tensor  # shape: [1,1,D,H,W]

def fft3d(image_tensor):
    image_complex = image_tensor.to(torch.complex64)
    kspace = torch.fft.fftn(image_complex, dim=(-3, -2, -1))
    return kspace  # shape: [1,1,D,H,W] complex

def reconstruct_kspace_from_half_3d(kspace):
    """
    Use half of the k-space volume and reconstruct the rest using Hermitian symmetry.
    Assumes kspace shape: [1, 1, D, H, W], torch.complex64
    """
    assert kspace.ndim == 5 and kspace.shape[1] == 1
    B, C, D, H, W = kspace.shape

    kspace_recon = torch.zeros_like(kspace)

    # Generate full 3D grid
    z = torch.arange(D)
    y = torch.arange(H)
    x = torch.arange(W)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Mask for positive-frequency half-space (includes DC plane)
    mask = (
        (zz < D // 2)
        | ((zz == D // 2) & (yy < H // 2))
        | ((zz == D // 2) & (yy == H // 2) & (xx <= W // 2))
    )

    # Copy only half of k-space
    kspace_recon[0, 0][mask] = kspace[0, 0][mask]

    # Hermitian conjugate completion
    zz_c = (-zz) % D
    yy_c = (-yy) % H
    xx_c = (-xx) % W

    # Fill the other half from conjugate
    mask_c = ~mask
    kspace_recon[0, 0][mask_c] = torch.conj(kspace[0, 0][zz_c[mask_c], yy_c[mask_c], xx_c[mask_c]])

    return kspace_recon

def recover_image_from_kspace(kspace):
    image_recon = torch.fft.ifftn(kspace, dim=(-3,-2,-1)).real  # only real part
    return image_recon

nifti_path = "/leelabsg/data/20252_unzip/1000502_20252_2_0/T1/T1_brain_to_MNI.nii.gz"  # 여기에 실제 경로 넣기
image = load_nifti_to_tensor(nifti_path)
kspace_ = fft3d(image)

img_orig = image[0,0]  # original real-valued image
k_recon = reconstruct_kspace_from_half_3d(kspace_)
img_recon = recover_image_from_kspace(k_recon)[0,0]
img_orig_np = img_orig.cpu().numpy()
img_recon_np = img_recon.cpu().numpy()

print("PSNR:", psnr(img_orig_np, img_recon_np, data_range=img_orig_np.max()))
print("SSIM:", ssim(img_orig_np, img_recon_np, data_range=img_orig_np.max()))


kspace_damaged = torch.zeros_like(kspace_)
kspace_damaged[0, 0][mask] = kspace_[0, 0][mask]
img_damaged = recover_image_from_kspace(kspace_damaged)[0, 0]
print("PSNR (without Hermitian reconstruction):", psnr(img_orig_np, img_damaged.cpu().numpy(), data_range=img_orig_np.max()))


def show_middle_slices(vol, title=""):
    mid_z, mid_y, mid_x = [s // 2 for s in vol.shape]
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    plt.subplot(1, 3, 1)
    plt.imshow(vol[mid_z, :, :], cmap='gray')
    plt.title("Axial")
    plt.subplot(1, 3, 2)
    plt.imshow(vol[:, mid_y, :], cmap='gray')
    plt.title("Coronal")
    plt.subplot(1, 3, 3)
    plt.imshow(vol[:, :, mid_x], cmap='gray')
    plt.title("Sagittal")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()

show_middle_slices(img_orig_np, "Original")
show_middle_slices(img_damaged.cpu().numpy(), "Damaged (Half only)")
show_middle_slices(img_recon_np, "Reconstructed (Hermitian)")

# Expandable to more datasets
datasets = {
    1: {
        "data_name": "T1_brain_to_MNI",
        "train_files": train_files,
        "val_files": valid_files,
        "modality": "mri",
    }
}

# Build training dataset
# Initialize file lists
train_files = {"mri": []}
val_files = {"mri": []}

# Function to add assigned class to datalist
def add_assigned_class_to_datalist(datalist, classname):
    for item in datalist:
        item["class"] = classname
    return datalist

# Process datasets
for _, dataset in datasets.items():
    train_files_i = dataset["train_files"]
    val_files_i = dataset["val_files"]
    print(f"{dataset['data_name']}: number of training data is {len(train_files_i)}.")
    print(f"{dataset['data_name']}: number of val data is {len(val_files_i)}.")

    # attach modality to each file
    modality = dataset["modality"]
    train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
    val_files[modality] += add_assigned_class_to_datalist(val_files_i, modality)

# Print total numbers for each modality
for modality in train_files.keys():
    print(f"Total number of training data for {modality} is {len(train_files[modality])}.")
    print(f"Total number of val data for {modality} is {len(val_files[modality])}.")

# Combine the data
train_files_combined = train_files["mri"]
val_files_combined = val_files["mri"]