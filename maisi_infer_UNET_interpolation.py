from __future__ import annotations

import argparse
import logging
import os
import json
import glob
import random
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import monai
from monai.data import DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from monai.transforms import Resize, Compose, EnsureChannelFirst, Spacing
from monai.transforms import Compose
from monai.utils import first
from tqdm import tqdm
import matplotlib.pyplot as plt

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.sample import ReconModel, check_input
from scripts.utils import define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot

def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data["training"]
    return [_item["image"].split("/")[4][:7] + "_emb.nii.gz" for _item in filenames_train] ###

def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate stable scale factor over multiple batches.
    """
    stds = []
    num_batches = 10

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        images = batch["image"].to(device)
        stds.append(torch.std(images, unbiased=False))  # More stable for batch use

    mean_std = torch.stack(stds).mean()
    scale_factor = 1.0 / mean_std

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)

    logger.info(f"Computed avg std over {num_batches} batches: {mean_std.item():.4f}")
    logger.info(f"Final scale_factor: {scale_factor.item():.4f}")
    return scale_factor

def prepare_data(
    train_files: list,
    device: torch.device,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
) -> DataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        include_body_region (bool): Whether to include body region in data

    Returns:
        DataLoader: Data loader for training.
    """

    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    train_transforms_list = [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="cond", func=lambda x: _load_data_from_file(x, "cond")),
    ]
    if include_body_region:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index", func=lambda x: _load_data_from_file(x, "top_region_index")
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: _load_data_from_file(x, "bottom_region_index")
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return DataLoader(train_ds, num_workers=4, batch_size=batch_size, shuffle=True)

def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed

def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder["autoencoder"]) ###
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor

def prepare_tensors(args: argparse.Namespace, device: torch.device) -> tuple:
    """
    Prepare necessary tensors for inference.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load tensors on.

    Returns:
        tuple: Prepared top_region_index_tensor, bottom_region_index_tensor, and spacing_tensor.
    """
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones(
        (len(spacing_tensor)), dtype=torch.long
    ).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor

def run_inference(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
    i: int #####
) -> np.ndarray:
    global _global_noise1, _global_noise2
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        modality_tensor (torch.Tensor): Modality tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # noise = torch.randn(
    #     (
    #         1,
    #         args.latent_channels,
    #         output_size[0] // divisor,
    #         output_size[1] // divisor,
    #         output_size[2] // divisor,
    #     ),
    #     device=device,
    # )
    g = torch.Generator(device=device).manual_seed(42)
    noise1 = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device, generator=g
    )
    noise2 = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device, generator=g
    )
    
    alpha = i / 9
    noise = (1 - alpha) * noise1 + alpha * noise2

    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )
    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }

            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )

            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = unet(**unet_inputs)
            if not isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image)  # type: ignore
            else:
                image, _ = noise_scheduler.step(model_output, t, image, next_t)  # type: ignore

        print(f"[latent] min={image.min().item():.3f}, max={image.max().item():.3f}, mean={image.mean().item():.3f}, std={image.std().item():.3f}") ###
        inferer = SlidingWindowInferer(
            roi_size=[64, 64, 64], # [80, 80, 80]
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.25, # 0.4
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)
        data = synthetic_images.squeeze().cpu().detach().numpy() #.astype(np.float32) ###
        # print(f"[decoded] dtype={data.dtype}, shape={data.shape}")
        # print(f"min={data.min()}, max={data.max()}, mean={data.mean()}")
        # try:
        #     std_safe = np.sqrt(np.mean((data - data.mean()) ** 2))
        #     print(f"std_safe: {std_safe}")
        # except Exception as e:
        #     print(f"std_safe computation failed: {e}")
        # a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        # data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        # print(f"[final before clip] min={data.min().item()}, max={data.max().item()}, mean={data.mean().item()}, std={data.std().item()}")
        # data = np.clip(data, a_min, a_max)
        # Float32 변환은 이미 적용됐다고 가정
        # 더 좁은 범위의 클리핑
        # vmin, vmax = np.percentile(data, [2, 98])  # or even [5, 95] 실험해도 좋아
        # print(f"[rescale] vmin={vmin:.2f}, vmax={vmax:.2f}")

        # # 먼저 클립
        # data = np.clip(data, vmin, vmax)
        # # MRI 기준으로 dynamic range를 줄이기 (예: [-500, 500])
        # a_min, a_max = -500, 500
        # data = (data - vmin) / (vmax - vmin) * (a_max - a_min) + a_min
        # a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        # data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        # data = np.clip(data, a_min, a_max)
        v_min, v_max = np.percentile(data, 0), np.percentile(data, 99.5)
        b_min, b_max = 0.0, 1.0
        if v_max == v_min:
            return np.zeros_like(data, dtype=np.float32)
        normed = (data - v_min) / (v_max - v_min)
        normed = normed * (b_max - b_min) + b_min
        return np.float32(data) # np.int16(data)

def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    logger: logging.Logger,
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
    logger.info(f"Saved {output_path}.")

@torch.inference_mode()
def diff_model_infer(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, i: int) -> None:
    """
    Main function to run the diffusion model inference.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    output_prefix = args.output_prefix
    ckpt_filepath = f"{args.model_dir}/{args.model_filename}"

    if local_rank == 0:
        logger.info(f"[config] ckpt_filepath -> {ckpt_filepath}.")
        logger.info(f"[config] random_seed -> {random_seed}.")
        logger.info(f"[config] output_prefix -> {output_prefix}.")
        logger.info(f"[config] output_size -> {output_size}.")
        logger.info(f"[config] out_spacing -> {out_spacing}.")

    #check_input(None, None, None, output_size, out_spacing, None) ###

    autoencoder, unet, scale_factor = load_models(args, device, logger) # scale_factor

    ### recalculate scale factor
    # include_body_region = unet.include_top_region_index_input
    # filenames_train = load_filenames(args.json_data_list)
    # if local_rank == 0:
    #     logger.info(f"num_files_train: {len(filenames_train)}")

    # train_files = []
    # for _i in range(len(filenames_train)):
    #     str_img = os.path.join(args.embedding_base_dir, filenames_train[_i])
    #     if not os.path.exists(str_img):
    #         continue

    #     str_info = os.path.join(args.embedding_base_dir, filenames_train[_i]) + ".json"
    #     train_files_i = {"image": str_img, "spacing": str_info, "cond": str_info} ###
    #     if include_body_region:
    #         train_files_i["top_region_index"] = str_info
    #         train_files_i["bottom_region_index"] = str_info
    #     train_files.append(train_files_i)
    # if dist.is_initialized():
    #     train_files = partition_dataset(
    #         data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
    #     )[local_rank]

    # train_loader = prepare_data(
    #     train_files, 
    #     device, 
    #     args.diffusion_unet_train["cache_rate"], 
    #     batch_size=128,
    #     include_body_region=include_body_region,
    # )
    # scale_factor = calculate_scale_factor(train_loader, device, logger)
    ###

    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(args, device)
    data = run_inference(
        args,
        device,
        autoencoder,
        unet,
        scale_factor,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
        output_size,
        divisor,
        logger,
        i
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = "{0}/{1}_seed{2}_size{3:d}x{4:d}x{5:d}_spacing{6:.2f}x{7:.2f}x{8:.2f}_{9}_rank{10}_{11:02d}_int.nii.gz".format(
        args.output_dir,
        output_prefix,
        random_seed,
        output_size[0],
        output_size[1],
        output_size[2],
        out_spacing[0],
        out_spacing[1],
        out_spacing[2],
        timestamp,
        local_rank,
        i
    )
    save_image(data, output_size, out_spacing, output_path, logger)

    if dist.is_initialized():
        dist.destroy_process_group()

    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference",
    )

    args = parser.parse_args()

    for i in range(10):
        data = diff_model_infer(args.env_config, args.train_config, args.model_def, args.num_gpus, i)
    # print(f"[final after clip]: {data.min()}, max: {data.max()}, mean: {data.mean()}, std: {data.std()}")

    # img = nib.load("/leelabsg/data/20252_unzip/3651056_20252_2_0/T1/T1_brain_to_MNI.nii.gz")
    # data = img.get_fdata()
    # print(f"[Real] min: {data.min()}, max: {data.max()}, mean: {data.mean()}, std: {data.std()}")

    args = load_config(args.env_config, args.train_config, args.model_def)
    input_dir = args.output_dir
    output_dir = os.path.join(args.output_dir, "slices")
    os.makedirs(output_dir, exist_ok=True)

    # 변환 파이프라인 (spacing + resize)
    target_spacing = (1.0, 1.0, 1.0)
    target_shape = (182, 218, 182)

    transform = Compose([
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
    nii_paths = glob.glob(os.path.join(input_dir, "*int.nii.gz"))
    print(f"Found {len(nii_paths)} files.")

    for nii_path in nii_paths:
        # Load and preprocess
        tensor = load_nii_as_tensor(nii_path)  # (1, H, W, D)
        tensor = transform(tensor)
        base_name = os.path.basename(nii_path).replace(".nii.gz", "")
        save_wandb_style_xyz_plot(tensor, base_name, output_dir)

    print("✅ Done.")