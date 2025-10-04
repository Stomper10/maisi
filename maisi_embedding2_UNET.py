from __future__ import annotations

import os
import json
import yaml
import logging
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist

import monai
from monai.config import print_config
from monai.transforms import Compose

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.utils import define_instance

print_config()

def create_transforms(dim: tuple = None) -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.

    Returns:
        Compose: Composed MONAI transforms.
    """
    if dim:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
                monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
                monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False), ### for MRI
                # monai.transforms.ScaleIntensityRanged( # for CT
                #     keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
                # ),
                monai.transforms.Resized(keys="image", spatial_size=dim, mode="trilinear"),
            ]
        )
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            ]
        )

def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)

def load_filenames(data_list_path: str, type: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_raw = json_data[type]
    return [_item["image"] for _item in filenames_raw]

def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    plain_transforms: Compose,
    new_transforms: Compose,
    logger: logging.Logger,
) -> None:
    """
    Process a single file to create training data.

    Args:
        filepath (str): Path to the file to be processed.
        args (argparse.Namespace): Configuration arguments.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to process the file on.
        plain_transforms (Compose): Plain transforms.
        new_transforms (Compose): New transforms.
        logger (logging.Logger): Logger for logging information.
    """
    out_filename_base = filepath.replace(".gz", "").replace(".nii", "")
    #out_filename_base = os.path.join(args.embedding_base_dir, out_filename_base)
    #out_filename = out_filename_base + "_emb.nii.gz"
    out_filename = os.path.join(args.embedding_base_dir, out_filename_base.split("/")[4][:7]) + "_emb.nii.gz" ###

    if os.path.isfile(out_filename):
        return

    test_data = {"image": os.path.join(args.data_base_dir, filepath)}
    transformed_data = plain_transforms(test_data)
    nda = transformed_data["image"]

    dim = [int(nda.meta["dim"][_i]) for _i in range(1, 4)]
    spacing = [float(nda.meta["pixdim"][_i]) for _i in range(1, 4)]

    logger.info(f"old dim: {dim}, old spacing: {spacing}")

    new_data = new_transforms(test_data)
    nda_image = new_data["image"]

    new_affine = nda_image.meta["affine"].numpy()
    nda_image = nda_image.numpy().squeeze()

    logger.info(f"new dim: {nda_image.shape}, new affine: {new_affine}")

    try:
        out_path = Path(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"out_filename: {out_filename}")

        with torch.amp.autocast("cuda"):
            pt_nda = torch.from_numpy(nda_image).float().to(device).unsqueeze(0).unsqueeze(0)
            z = autoencoder.encode_stage_2_inputs(pt_nda)
            logger.info(f"z: {z.size()}, {z.dtype}")

            out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
            nib.save(out_img, out_filename)
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed() ### num_gpus=num_gpus
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder["autoencoder"]) ###
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    # training files
    filenames_raw = load_filenames(args.json_data_list, "training")
    logger.info(f"filenames_raw: {filenames_raw}")

    plain_transforms = create_transforms(dim=None)

    for _iter in range(len(filenames_raw)):
        if _iter % world_size != local_rank:
            continue

        filepath = filenames_raw[_iter]
        new_dim = tuple(
            round_number(
                int(plain_transforms({"image": os.path.join(args.data_base_dir, filepath)})["image"].meta["dim"][_i])
            )
            for _i in range(1, 4)
        )
        new_transforms = create_transforms(new_dim)

        process_file(filepath, args, autoencoder, device, plain_transforms, new_transforms, logger)

    # validation files
    val_filenames_raw = load_filenames(args.val_json_data_list, "validation")
    logger.info(f"val_filenames_raw: {val_filenames_raw}")

    plain_transforms = create_transforms(dim=None)

    for _iter in range(len(val_filenames_raw)):
        if _iter % world_size != local_rank:
            continue

        val_filepath = val_filenames_raw[_iter]
        new_dim = tuple(
            round_number(
                int(plain_transforms({"image": os.path.join(args.data_base_dir, val_filepath)})["image"].meta["dim"][_i])
            )
            for _i in range(1, 4)
        )
        new_transforms = create_transforms(new_dim)

        process_file(val_filepath, args, autoencoder, device, plain_transforms, new_transforms, logger)

    if dist.is_initialized():
        dist.destroy_process_group()

# Create .json files
def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files

def create_json_files(gz_files, df):
    """Create .json files for each .gz file with the specified keys and values."""
    for gz_file in gz_files:
        # Load the NIfTI image
        img = nib.load(gz_file)

        # Get the dimensions and spacing
        dimensions = img.shape
        dimensions = dimensions[:3]
        spacing = img.header.get_zooms()[:3]
        spacing = spacing[:3]
        spacing = [float(_item) for _item in spacing]

        # Create the dictionary with the specified keys and values
        # The region can be selected from one of four regions from top to bottom.
        # [1,0,0,0] is the head and neck, [0,1,0,0] is the chest region, [0,0,1,0]
        # is the abdomen region, and [0,0,0,1] is the lower body region.
        data = [
            {
                "dim": dimensions,
                "spacing": spacing,
                "top_region_index": [0, 0, 0, 0], ###
                "bottom_region_index": [0, 0, 0, 0], ###
                "cond": [
                    df.loc[df['eid'] == int(gz_file.split("/")[-1][:7]), "norm_age"].values[0],
                    df.loc[df['eid'] == int(gz_file.split("/")[-1][:7]), "sex"].values[0],
                    df.loc[df['eid'] == int(gz_file.split("/")[-1][:7]), "norm_csf"].values[0],
                    df.loc[df['eid'] == int(gz_file.split("/")[-1][:7]), "norm_cgm"].values[0],
                    df.loc[df['eid'] == int(gz_file.split("/")[-1][:7]), "norm_dgm"].values[0],
                    df.loc[df['eid'] == int(gz_file.split("/")[-1][:7]), "norm_wm"].values[0],
                ],
            }
        ]
        #logger.info(f"data: {data[0]}.")

        # Create the .json filename
        json_filename = gz_file + ".json"

        # Write the dictionary to the .json file
        with open(json_filename, "w") as json_file:
            json.dump(data[0], json_file, indent=4) ###
        print(f"Save json file to {json_filename}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for distributed training")
    parser.add_argument(
        "--train_label_dir",
        type=str,
        default=None,
        help="A training label dir.",
    ),
    parser.add_argument(
        "--valid_label_dir",
        type=str,
        default=None,
        help="A validation label dir.",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    diff_model_create_training_data(args.env_config, args.train_config, args.model_def, args.num_gpus)

    df = pd.read_csv(args.train_label_dir)
    df["dgm"] = df["p25005_i2"] - df["p25001_i2"]
    df["norm_age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
    df["norm_csf"] = (df["p25003_i2"] - df["p25003_i2"].min()) / (df["p25003_i2"].max() - df["p25003_i2"].min())
    df["norm_cgm"] = (df["p25001_i2"] - df["p25001_i2"].min()) / (df["p25001_i2"].max() - df["p25001_i2"].min())
    df["norm_dgm"] = (df["dgm"] - df["dgm"].min()) / (df["dgm"].max() - df["dgm"].min())
    df["norm_wm"] = (df["p25007_i2"] - df["p25007_i2"].min()) / (df["p25007_i2"].max() - df["p25007_i2"].min())
    df_train = df
    
    df = pd.read_csv(args.valid_label_dir)
    df["dgm"] = df["p25005_i2"] - df["p25001_i2"]
    df["norm_age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
    df["norm_csf"] = (df["p25003_i2"] - df["p25003_i2"].min()) / (df["p25003_i2"].max() - df["p25003_i2"].min())
    df["norm_cgm"] = (df["p25001_i2"] - df["p25001_i2"].min()) / (df["p25001_i2"].max() - df["p25001_i2"].min())
    df["norm_dgm"] = (df["dgm"] - df["dgm"].min()) / (df["dgm"].max() - df["dgm"].min())
    df["norm_wm"] = (df["p25007_i2"] - df["p25007_i2"].min()) / (df["p25007_i2"].max() - df["p25007_i2"].min())
    df_valid = df

    df_all = pd.concat([df_train, df_valid], ignore_index=True)

    with open(args.env_config, "r") as f:
        env_config_out = json.load(f)

    folder_path = env_config_out["embedding_base_dir"]
    gz_files = list_gz_files(folder_path)
    create_json_files(gz_files, df_all)



if __name__ == "__main__":
    main()