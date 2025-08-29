from __future__ import annotations

import argparse
import logging
import os
import sys
import json
import random
from datetime import datetime
import numpy as np
import nibabel as nib

import torch
import torch.distributed as dist
import monai
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage
from monai.utils import RankFilter
from monai.transforms import Resize, Compose, EnsureChannelFirst, Spacing
from monai.transforms import Compose
import matplotlib.pyplot as plt

from scripts.sample import check_input, ldm_conditional_sample_one_image
from scripts.utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from scripts.utils_plot import find_label_center_loc, get_xyz_plot



@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.infer")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_controlnet_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="./configs/config_maisi_controlnet_train.json",
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
    
    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.infer")
    # whether to use distributed data parallel
    use_ddp = args.num_gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    with open(args.env_config, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.model_def, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.train_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)
    
    # Step 1: set data loader
    # with open(args.valid_files_ctrl, 'r') as f:
    #     valid_data = json.load(f)

    # for item in valid_data['validation']:
    #     original_filename = os.path.basename(item['image'])
    #     new_filename = original_filename.replace('.nii.gz', '_emb.nii.gz')
    #     new_image_path = os.path.join(embedding_dir, new_filename)
    #     item['image'] = new_image_path
    #     item['dim'] = [256, 256, 128]
    #     item['spacing'] = [0.9375, 0.9375, 1.2109375]
    #     item['fold'] = 0

    #     if include_body_region:
    #         item["top_region_index"] = [0, 1, 0, 0]
    #         item["bottom_region_index"] = [0, 0, 0, 1]
    # Data
    _, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list_ctrl_val, ###
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
    )
    
    # Step 2: define AE, diffusion model and controlnet
    # define AE
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # load trained autoencoder model
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")
        autoencoder_ckpt = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(autoencoder_ckpt["autoencoder"])
        logger.info(f"Load trained diffusion model from {args.trained_autoencoder_path}.")
    else:
        logger.info("trained autoencoder model is not loaded.")

    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    
    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        logger.info("trained diffusion model is not loaded.")
        scale_factor = 1.0
        logger.info(f"set scale_factor -> {scale_factor}.")
    
    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)["ctrl_state_dict"] ###
        )
        logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else:
        logger.info("trained controlnet is not loaded.")

    noise_scheduler = define_instance(args, "noise_scheduler")
    
    # Step 3: inference
    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    for batch in val_loader:
        # get label mask
        labels = batch["label"].to(device)
        # get corresponding conditions
        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        else:
            top_region_index_tensor = None
            bottom_region_index_tensor = None
        spacing_tensor = batch["spacing"].to(device)
        if include_modality:
            # modality_tensor = args.controlnet_infer["modality"] * torch.ones((len(labels),), dtype=torch.long).to(device)
            random_modality = random.randint(0, 3) ###
            modality_tensor = torch.tensor([random_modality], dtype=torch.long).to(device) ###
        else:
            modality_tensor = None
        out_spacing = tuple((batch["spacing"].squeeze().numpy() / 100).tolist())
        # get target dimension
        dim = batch["dim"]
        output_size = (dim[0].item(), dim[1].item(), dim[2].item())
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
        # check if output_size and out_spacing are valid.
        #check_input(None, None, None, output_size, out_spacing, None)
        # generate a single synthetic image using a latent diffusion model with controlnet.
        synthetic_images, _ = ldm_conditional_sample_one_image(
            autoencoder=autoencoder,
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            device=device,
            combine_label_or=labels,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.controlnet_infer["num_inference_steps"],
            autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
            autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
        )
        print(synthetic_images.shape, flush=True)
        print(type(synthetic_images), flush=True)
        # save image/label pairs
        labels = decollate_batch(batch)[0]["label"]
        output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        labels.meta["filename_or_obj"] = "sample.nii.gz"
        synthetic_images = MetaTensor(synthetic_images.squeeze(0), meta=labels.meta)
        img_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_image",
            separate_folder=False,
        )
        img_saver(synthetic_images)
        label_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_label",
            separate_folder=False,
        )
        label_saver(labels)
    if use_ddp:
        dist.destroy_process_group()

    # img = nib.load("/leelabsg/data/20252_unzip/3651056_20252_2_0/T1/T1_brain_to_MNI.nii.gz")
    # data = img.get_fdata()
    # print(f"[Real] min: {data.min()}, max: {data.max()}, mean: {data.mean()}, std: {data.std()}")

    # args = load_config(args.env_config, args.train_config, args.model_def)
    # input_dir = args.output_dir
    # output_dir = os.path.join(args.output_dir, "slices")
    # os.makedirs(output_dir, exist_ok=True)

    # 변환 파이프라인 (spacing + resize)
    target_spacing = (1.0, 1.0, 1.0)
    target_shape = (240, 240, 155)

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
    nii_paths = [nii for nii in os.listdir(args.output_dir) if "nii.gz" in nii]
    #nii_paths = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    print(f"Found {len(nii_paths)} files.")

    for nii_path in nii_paths:
        # Load and preprocess
        tensor = load_nii_as_tensor(os.path.join(args.output_dir, nii_path))  # (1, H, W, D)
        tensor = transform(tensor)
        base_name = nii_path.replace(".nii.gz", "")
        save_wandb_style_xyz_plot(tensor, base_name, args.output_dir)

    print("✅ Done.")
        
if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
