import os
import copy
import yaml
import json
import argparse
from monai.config import print_config

print_config()

MODALITY_SUFFIX = {
    "t1n": 0,
    "t1c": 1,
    "t2w": 2,
    "t2f": 3,
}

def build_items_single_mod(data_dir, type): # "/leelabsg/data/BraTS25/.../train"
    items = []

    for case in sorted(os.listdir(data_dir)):
        case_dir = os.path.join(data_dir, case)
        if not os.path.isdir(case_dir):
            continue
        seg_path = os.path.join(case_dir, f"{case}-seg.nii.gz")
        if not os.path.exists(seg_path):
            continue

        for mod, cls in MODALITY_SUFFIX.items():
            img_path = os.path.join(case_dir, f"{case}-{mod}.nii.gz")
            if os.path.exists(img_path):
                items.append({
                    "image": img_path,
                    "label": seg_path,
                    "modality_class": cls,   # 0=t1n, 1=t1c, 2=t2w, 3=t2f
                    "case_id": case,
                    "modality": mod,
                })

    return {type: items}

def get_run_name(manual_name=None, default_prefix="manual"):
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME")
    if manual_name:
        return manual_name
    elif job_id and job_name:
        return f"{job_name}_{job_id}"
    elif job_id:
        return f"slurm_{job_id}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{default_prefix}_{timestamp}"

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vae_path", type=str, default=None, help="pretrained_vae_path")
    parser.add_argument("--model_def_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow_brats.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_diff_model_brats.json")
    parser.add_argument("--env_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/environment_maisi_diff_model.json")
    parser.add_argument("--train_data_dir", type=str, default=None, help="train data path")
    parser.add_argument("--valid_data_dir", type=str, default=None, help="valid data path")
    parser.add_argument("--embedding_dir", type=str, default="/leelabsg/data/ex_MAISI")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    args.run_name = get_run_name(args.run_name)
    return args

def main():
    args = load_config()
    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    with open(args.model_def_path, "r") as f:
        model_def = json.load(f)
    
    maisi_version = args.model_def_path.split("/")[-1]
    include_body_region = model_def["include_body_region"]
    print(f"MAISI version is {maisi_version}, whether to use body_region is {include_body_region}")
    train_files = build_items_single_mod(args.train_data_dir, "training")
    valid_files = build_items_single_mod(args.valid_data_dir, "validation")
    
    dataroot_dir = ""
    work_dir = os.path.join(args.embedding_dir, args.run_name)
    os.makedirs(work_dir, exist_ok=True)

    datalist_file = os.path.join(work_dir, "train_files.json")
    with open(datalist_file, "w") as f:
        json.dump(train_files, f)
    val_datalist_file = os.path.join(work_dir, "valid_files.json")
    with open(val_datalist_file, "w") as f:
        json.dump(valid_files, f)

    # Setup
    env_config_path = args.env_config_path
    train_config_path = args.train_config_path
    model_def_path = args.model_def_path

    # Load environment configuration, model configuration and model definition
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    with open(train_config_path, "r") as f:
        train_config = json.load(f)

    with open(model_def_path, "r") as f:
        model_def = json.load(f)

    env_config_out = copy.deepcopy(env_config)
    train_config_out = copy.deepcopy(train_config)
    model_def_out = copy.deepcopy(model_def)

    env_config_out["data_base_dir"] = dataroot_dir
    env_config_out["embedding_base_dir"] = os.path.normpath(os.path.join(work_dir, env_config_out["embedding_base_dir"]))
    env_config_out["json_data_list"] = datalist_file
    env_config_out["val_json_data_list"] = val_datalist_file
    env_config_out["model_dir"] = os.path.normpath(os.path.join(work_dir, env_config_out["model_dir"]))
    env_config_out["output_dir"] = os.path.normpath(os.path.join(work_dir, env_config_out["output_dir"]))
    env_config_out["trained_autoencoder_path"] = args.pretrained_vae_path #.pth

    # Create necessary directories
    os.makedirs(env_config_out["embedding_base_dir"], exist_ok=True)
    os.makedirs(env_config_out["model_dir"], exist_ok=True)
    os.makedirs(env_config_out["output_dir"], exist_ok=True)

    env_config_filepath = os.path.join(work_dir, "environment_maisi_diff_model.json")
    with open(env_config_filepath, "w") as f:
        json.dump(env_config_out, f, sort_keys=True, indent=4)

    # Update model configuration for demo
    train_config_out["diffusion_unet_train"]["batch_size"] = 8
    train_config_out["diffusion_unet_train"]["val_batch_size"] = 4
    train_config_out["diffusion_unet_train"]["cache_rate"] = 0.1
    train_config_out["diffusion_unet_train"]["lr"] = 0.0001
    train_config_out["diffusion_unet_train"]["seed"] = 42
    train_config_out["diffusion_unet_train"]["report_to"] = True
    train_config_out["diffusion_unet_train"]["max_train_steps"] = 50000
    train_config_out["diffusion_unet_train"]["num_valid"] = 1000
    train_config_out["diffusion_unet_train"]["checkpointing_steps"] = 5000
    train_config_out["diffusion_unet_train"]["validation_steps"] = 5000
    train_config_out["diffusion_unet_inference"]["dim"] = [256,256,128]
    train_config_out["diffusion_unet_inference"]["spacing"] = [0.9375,0.9375,1.2109375]

    train_config_filepath = os.path.join(work_dir, "config_maisi_diff_model.json")
    with open(train_config_filepath, "w") as f:
        json.dump(train_config_out, f, sort_keys=True, indent=4)

    # Update model definition for demo
    model_def_out["autoencoder_def"]["num_splits"] = 4
    model_def_out["diffusion_unet_def"]["num_class_embeds"] = 4
    model_def_filepath = os.path.join(work_dir, "config_maisi.json")
    with open(model_def_filepath, "w") as f:
        json.dump(model_def_out, f, sort_keys=True, indent=4)

    # Print files and folders under work_dir
    print(f"files and folders under {work_dir}: {os.listdir(work_dir)}.")

    # Adjust based on the number of GPUs you want to use
    num_gpus = 1
    print(f"number of GPUs: {num_gpus}.")



if __name__ == "__main__":
    main()