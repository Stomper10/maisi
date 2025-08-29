import os
import copy
import yaml
import json
import argparse
from monai.config import print_config

print_config()

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
    parser.add_argument("--pretrained_diff_path", type=str, default=None, help="pretrained_diff_path")
    #parser.add_argument("--pretrained_control_path", type=str, default=None, help="pretrained_control_path")
    parser.add_argument("--model_def_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_controlnet_train.json")
    parser.add_argument("--env_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/environment_maisi_controlnet_train.json")
    parser.add_argument("--label_dir", type=str, default=None, help="train_mask_label_path")
    parser.add_argument("--embedding_dir", type=str, default="/leelabsg/data/ex_MAISI")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    args.run_name = get_run_name(args.run_name)
    return args


# logger = setup_logging("notebook")


def main():
    args = load_config()
    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    with open(args.model_def_path, "r") as f:
        model_def = json.load(f)
    
    maisi_version = args.model_def_path.split("/")[-1]
    include_body_region = model_def["include_body_region"]
    print(f"MAISI version is {maisi_version}, whether to use body_region is {include_body_region}")

    # Data
    dataroot_dir = ""
    work_dir = os.path.join(args.embedding_dir, args.run_name)
    embedding_dir = os.path.join(work_dir, "embeddings")
    
    # train_files = {
    #     "training": [
    #         {
    #             "image": os.path.join(embedding_dir, emb_name),
    #             "label": os.path.join(
    #                 args.label_dir,
    #                 emb_name.replace("_emb.nii.gz", "").rsplit("-", 1)[0],  # 디렉토리 이름
    #                 f"{emb_name.replace('_emb.nii.gz', '').rsplit('-', 1)[0]}-seg.nii.gz"  # 파일 이름
    #             ),
    #             "fold": 1,
    #             "dim": [256, 256, 128], ### 변환된 dim
    #             "spacing": [0.9375, 0.9375, 1.2109375], ### 변환된 spacing
    #         }
    #         for emb_name in os.listdir(embedding_dir)
    #         if emb_name.endswith(".nii.gz")
    #     ]
    # }

    # training
    unet_json_data_list = os.path.join(embedding_dir, args.run_name, "train_files.json")
    with open(unet_json_data_list, 'r') as f:
        train_data = json.load(f)

    for item in train_data['training']:
        original_filename = os.path.basename(item['image'])
        new_filename = original_filename.replace('.nii.gz', '_emb.nii.gz')
        new_image_path = os.path.join(embedding_dir, new_filename)
        item['image'] = new_image_path
        item['dim'] = [256, 256, 128]
        item['spacing'] = [0.9375, 0.9375, 1.2109375]
        item['fold'] = 1

        if include_body_region:
            item["top_region_index"] = [0, 1, 0, 0]
            item["bottom_region_index"] = [0, 0, 0, 1]

    # validation
    val_unet_json_data_list = os.path.join(embedding_dir, args.run_name, "valid_files.json")
    with open(val_unet_json_data_list, 'r') as f:
        valid_data = json.load(f)

    for item in valid_data['validation']:
        original_filename = os.path.basename(item['image'])
        new_filename = original_filename.replace('.nii.gz', '_emb.nii.gz')
        new_image_path = os.path.join(embedding_dir, new_filename)
        item['image'] = new_image_path
        item['dim'] = [256, 256, 128]
        item['spacing'] = [0.9375, 0.9375, 1.2109375]
        item['fold'] = 0

        if include_body_region:
            item["top_region_index"] = [0, 1, 0, 0]
            item["bottom_region_index"] = [0, 0, 0, 1]

    train_data['training'].extend(valid_data['validation'])
    combined_datalist_file = os.path.join(work_dir, "train_files_ctrl.json")
    with open(combined_datalist_file, 'w') as f:
        json.dump(train_data, f, indent=4)

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
    #env_config_out["json_data_list"] = datalist_file
    env_config_out["json_data_list_ctrl"] = combined_datalist_file
    env_config_out["model_dir"] = os.path.join(work_dir, env_config_out["model_dir"])
    env_config_out["output_dir"] = os.path.join(work_dir, env_config_out["output_dir"])
    env_config_out["tfevent_path"] = os.path.join(work_dir, env_config_out["tfevent_path"])
    env_config_out["trained_autoencoder_path"] = args.pretrained_vae_path #.pth
    env_config_out["trained_diffusion_path"] = args.pretrained_diff_path
    #env_config_out["trained_controlnet_path"] = args.pretrained_control_path
    env_config_out["exp_name"] = "tutorial_training_example"

    # Create necessary directories
    os.makedirs(env_config_out["model_dir"], exist_ok=True)
    os.makedirs(env_config_out["output_dir"], exist_ok=True)
    os.makedirs(env_config_out["tfevent_path"], exist_ok=True)

    env_config_filepath = os.path.join(work_dir, "environment_maisi_controlnet_train.json")
    with open(env_config_filepath, "w") as f:
        json.dump(env_config_out, f, sort_keys=True, indent=4)

    # Update model configuration for demo
    train_config_out["controlnet_train"]["n_epochs"] = 10000
    train_config_out["controlnet_train"]["weighted_loss"] = 1
    train_config_out["controlnet_train"]["weighted_loss_label"] = [None]
    train_config_out["controlnet_infer"]["num_inference_steps"] = 10

    train_config_filepath = os.path.join(work_dir, "config_maisi_controlnet_train.json")
    with open(train_config_filepath, "w") as f:
        json.dump(train_config_out, f, sort_keys=True, indent=4)

    # Update model definition for demo
    model_def_out["autoencoder_def"]["num_splits"] = 4
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