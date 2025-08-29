import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, Resized
from tqdm import tqdm
from scripts.transforms import VAE_Transform
from scipy.ndimage import mean

# --- 1. 주파수 영역 분석 함수 ---
def analyze_frequency_spectrum(image_tensor, device):
    """
    이미지 텐서(B, C, H, W, D)를 받아 평균 파워 스펙트럼을 계산
    """
    images_np = image_tensor.cpu().numpy()
    avg_power_spectrum = None # 배치 내 모든 이미지에 대해 파워 스펙트럼을 계산하고 누적
    
    for i in range(images_np.shape[0]):
        img = images_np[i, 0, :, :, :]
        
        # 3D 푸리에 변환 -> 중심 이동 -> 파워 스펙트럼 계산 (로그 스케일)
        f_transform = np.fft.fftn(img)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.log(1 + np.abs(f_shift)**2)
        
        if avg_power_spectrum is None:
            avg_power_spectrum = power_spectrum
        else:
            avg_power_spectrum += power_spectrum
            
    # 평균 파워 스펙트럼 계산
    avg_power_spectrum /= images_np.shape[0]
    return avg_power_spectrum

# --- 2. 잔차 맵 시각화 함수 ---
def visualize_residual_maps(gen_images, real_images_pool, num_maps=3):
    """
    생성된 이미지와 가장 유사한 실제 이미지를 찾아 잔차 맵을 시각화합니다.
    """
    gen_images_np = gen_images.cpu().numpy()
    real_images_pool_np = real_images_pool.cpu().numpy()
    
    # 시각화할 이미지 수만큼 반복
    for i in range(min(num_maps, gen_images_np.shape[0])):
        gen_img = gen_images_np[i, 0] # (H, W, D)
        
        # L1 거리(Mean Absolute Error)를 사용하여 가장 유사한 실제 이미지 탐색
        best_match_idx = -1
        min_distance = float('inf')
        
        for j in tqdm(range(real_images_pool_np.shape[0]), desc=f"Finding best match for Gen Img #{i+1}"):
            real_img = real_images_pool_np[j, 0]
            distance = np.mean(np.abs(gen_img - real_img))
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
                
        best_match_real_img = real_images_pool_np[best_match_idx, 0]
        
        # 잔차 맵 계산
        residual_map = np.abs(gen_img - best_match_real_img)
        
        # 결과 시각화 (3D 데이터의 중앙 슬라이스)
        slice_idx = gen_img.shape[2] // 2 # Z축 기준 중앙 슬라이스
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Residual Map Visualization for Generated Image #{i+1}', fontsize=16)
        
        axes[0].imshow(gen_img[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Generated Image')
        axes[0].axis('off')
        
        axes[1].imshow(best_match_real_img[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Closest Real Image (L1: {min_distance:.4f})')
        axes[1].axis('off')
        
        axes[2].imshow(residual_map[:, :, slice_idx], cmap='hot')
        axes[2].set_title('Residual Map (Difference)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"residual_map_{i+1}.png")
        print(f"Saved residual_map_{i+1}.png")
        plt.close()


if __name__ == '__main__':
    # =========================================================================
    # --- 설정 ---
    # =========================================================================
    # Function to add assigned class to datalist
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist
    
    REAL_LABEL_DIR = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv"
    REAL_IMAGE_DIR = "/leelabsg/data/20252_unzip/"
    FAKE_IMAGE_DIR = "/leelabsg/data/ex_MAISI/E0_UKB/recon/whole" # "/leelabsg/data/ex_MAISI/E0_UKB/predictions"

    real_train_label_df = pd.read_csv(REAL_LABEL_DIR)[0:1000]
    real_train_files = [{"image": os.path.join(REAL_IMAGE_DIR, image_name)} for image_name in real_train_label_df["rel_path"]]
    real_datasets = {
        1: {
            "data_name": "T1_brain_to_MNI",
            "train_files": real_train_files,
            "modality": "mri",
        }
    }
        
    fake_train_files = [{"image": os.path.join(FAKE_IMAGE_DIR, image_name)} for image_name in os.listdir(FAKE_IMAGE_DIR) if "nii.gz" in image_name][:1000]
    # fake_train_label_df = pd.read_csv(REAL_LABEL_DIR)[1000:2000] ###
    # fake_train_files = [{"image": os.path.join(REAL_IMAGE_DIR, image_name)} for image_name in fake_train_label_df["rel_path"]] ###
    fake_datasets = {
        1: {
            "data_name": "T1_brain_to_MNI",
            "train_files": fake_train_files,
            "modality": "mri",
        }
    }

    # Process datasets
    real_train_files = {"mri": []}
    for _, dataset in real_datasets.items():
        train_files_i = dataset["train_files"]
        print(f"{dataset['data_name']}: number of real training data is {len(train_files_i)}.")
        # attach modality to each file
        modality = dataset["modality"]
        real_train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
    
    for modality in real_train_files.keys():
        print(f"Total number of real training data for {modality} is {len(real_train_files[modality])}.")

    fake_train_files = {"mri": []}
    for _, dataset in fake_datasets.items():
        train_files_i = dataset["train_files"]
        print(f"{dataset['data_name']}: number of fake training data is {len(train_files_i)}.")
        # attach modality to each file
        modality = dataset["modality"]
        fake_train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
    
    for modality in fake_train_files.keys():
        print(f"Total number of fake training data for {modality} is {len(fake_train_files[modality])}.")

    real_train_files_combined = real_train_files["mri"]
    fake_train_files_combined = fake_train_files["mri"]
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

    real_dataset_train = CacheDataset(data=real_train_files_combined, transform=train_transform, cache_rate=0.0, num_workers=8)
    fake_dataset_train = CacheDataset(data=fake_train_files_combined, transform=train_transform, cache_rate=0.0, num_workers=8)

    BATCH_SIZE = 2
    NUM_SAMPLES_FOR_SPECTRUM = 64  # 스펙트럼 분석에 사용할 샘플 수
    NUM_SAMPLES_FOR_RESIDUAL = 3   # 잔차 맵을 생성할 샘플 수
    
    real_loader = DataLoader(real_dataset_train, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
    fake_loader = DataLoader(fake_dataset_train, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 주파수 분석 실행 ---
    print("--- Starting Frequency Spectrum Analysis ---")
    real_spectra, gen_spectra = [], []
    
    for i, batch in enumerate(real_loader):
        if i * BATCH_SIZE >= NUM_SAMPLES_FOR_SPECTRUM: break
        real_spectra.append(analyze_frequency_spectrum(batch["image"], device))
    
    for i, batch in enumerate(fake_loader):
        if i * BATCH_SIZE >= NUM_SAMPLES_FOR_SPECTRUM: break
        gen_spectra.append(analyze_frequency_spectrum(batch["image"], device))

    # 각 배치의 스펙트럼을 평균
    avg_real_spectrum = np.mean(real_spectra, axis=0)
    avg_gen_spectrum = np.mean(gen_spectra, axis=0)
    
    # 시각화 (중앙 슬라이스)
    slice_idx = avg_real_spectrum.shape[1] // 2 # Y축 기준
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(avg_real_spectrum[:, slice_idx, :], cmap='viridis')
    axes[0].set_title('Average Power Spectrum (Real Images)')
    axes[0].axis('off')
    axes[1].imshow(avg_gen_spectrum[:, slice_idx, :], cmap='viridis')
    axes[1].set_title('Average Power Spectrum (Generated Images)')
    axes[1].axis('off')
    plt.savefig("frequency_spectrum_comparison.png")
    print("Saved frequency_spectrum_comparison.png")
    plt.close()

    print("\n--- Starting 1D Radial Profile Analysis ---")

    def radial_profile(data):
        """2D 데이터의 방사형 프로파일을 계산합니다."""
        center_y, center_x = np.array(data.shape) // 2
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        # 각 반지름(r)에 해당하는 픽셀 값들의 평균을 계산합니다.
        # np.bincount를 사용하면 각 반지름에 몇 개의 픽셀이 있는지 셀 수 있습니다.
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        
        # 0으로 나누는 것을 방지
        radial_mean = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr!=0)
        return radial_mean

    # 3D 스펙트럼의 중앙 슬라이스를 가져와서 1D 프로파일 계산
    real_profile_1d = radial_profile(avg_real_spectrum[:, slice_idx, :])
    gen_profile_1d = radial_profile(avg_gen_spectrum[:, slice_idx, :])

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(real_profile_1d, label='Real Images', color='blue')
    plt.plot(gen_profile_1d, label='Generated Images', color='red', linestyle='--')

    plt.title('1D Radial Profile of Power Spectrum')
    plt.xlabel('Frequency (Distance from center)')
    plt.ylabel('Average Power (Log Scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(left=0) # X축 시작을 0으로 고정
    plt.savefig("frequency_profile_1d.png")
    print("Saved frequency_profile_1d.png")
    plt.close()

    # --- 2. 잔차 맵 시각화 실행 ---
    print("\n--- Starting Residual Map Visualization ---")
    # 시각화를 위한 이미지 로드 (생성 이미지와 검색 대상이 될 실제 이미지)
    gen_images_for_residual = next(iter(DataLoader(fake_dataset_train, batch_size=NUM_SAMPLES_FOR_RESIDUAL)))["image"]
    real_images_pool = next(iter(DataLoader(real_dataset_train, batch_size=100)))["image"] # 100개 중에서 탐색
    
    visualize_residual_maps(gen_images_for_residual, real_images_pool, num_maps=NUM_SAMPLES_FOR_RESIDUAL)
    
    print("\nAnalysis complete.")