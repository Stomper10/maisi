import os
import time
#import glob
import json
import yaml
import wandb
import argparse
#import tempfile
#from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import trange #, tqdm

import torch
from monai.networks.nets import PatchDiscriminator
#from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
#from torch.utils.tensorboard import SummaryWriter

from utils import count_parameters
from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot #, show_image

import warnings

warnings.filterwarnings("ignore")

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
    #parser.add_argument("--custom_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_custom.json")
    parser.add_argument("--model_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_vae_train.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    args.run_name = get_run_name(args.run_name)
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

def resume_from_latest(autoencoder,
                       discriminator,
                       optimizer_g, 
                       optimizer_d,
                       scheduler_g,
                       scheduler_d,
                       scaler_g,
                       scaler_d,
                       output_dir, device):
    if not os.path.exists(output_dir):
        print("[Resume] No checkpoint directory found. Starting fresh.")
        return 0

    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, d))
        and os.path.exists(os.path.join(output_dir, d, "model.pt"))
    ]
    if len(ckpts) == 0:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0

    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    latest_ckpt_dir = ckpts[-1]
    print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")
    
    path = os.path.join(output_dir, latest_ckpt_dir, "model.pt")
    ckpt = torch.load(path, map_location=device)
    
    autoencoder.load_state_dict(ckpt["autoencoder"])
    discriminator.load_state_dict(ckpt["discriminator"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])
    scheduler_g.load_state_dict(ckpt["scheduler_g"])
    scheduler_d.load_state_dict(ckpt["scheduler_d"])
    scaler_g.load_state_dict(ckpt["scaler_g"])
    scaler_d.load_state_dict(ckpt["scaler_d"])

    return ckpt.get("step", 0)



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

    run_name = args.run_name
    print("[Config] Loaded hyperparameters:")
    print(yaml.dump(args, sort_keys=False))

    if args.report_to:
        wandb.init(project="maisi", config=args, name=run_name)
        wandb.run.define_metric("step")
        wandb.run.define_metric("train_ae/loss", step_metric="step")
        wandb.run.define_metric("valid_ae/loss", step_metric="step")
        wandb.run.define_metric("train_ae/lr", step_metric="step")

    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    tfevent_dir = os.path.join(output_dir, "tfevent")
    os.makedirs(tfevent_dir, exist_ok=True)

    # Data
    train_label_df = pd.read_csv(args.train_label_dir)
    valid_label_df = pd.read_csv(args.valid_label_dir)

    train_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in train_label_df["rel_path"]]
    valid_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in valid_label_df["rel_path"]]

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

    train_transform = VAE_Transform(
        is_train=True,
        random_aug=args.random_aug,  # whether apply random data augmentation for training
        k=4,  # patches should be divisible by k
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=weight_dtype,  # final data type
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
        resolution=(182,218,182) ###
    )
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        val_patch_size=args.val_patch_size,  # if None, will validate on whole image volume
        output_dtype=weight_dtype,  # final data type
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
        resolution=(182,218,182) ###
    )

    # Build dataloader
    print(f"Total number of training data is {len(train_files_combined)}.")
    dataset_train = CacheDataset(data=train_files_combined, transform=train_transform, cache_rate=args.cache, num_workers=8)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    print(f"Total number of validation data is {len(val_files_combined)}.")
    dataset_val = CacheDataset(data=val_files_combined, transform=val_transform, cache_rate=args.cache, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=4, shuffle=True)

    # args.autoencoder_def["num_splits"] = 1
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)



    # Training config
    # config loss and loss weight
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
        print("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    loss_perceptual = (
        PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    )

    # config optimizer and lr scheduler
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)

    # please adjust the learning rate warmup rule based on your dataset and n_epochs
    # def warmup_rule(epoch):
    #     # learning rate warmup rule
    #     if epoch < 10:
    #         return 0.01
    #     elif epoch < 20:
    #         return 0.1
    #     else:
    #         return 1.0

    # scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    # scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)
    scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.max_train_steps)
    scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.max_train_steps)

    # set AMP scaler
    if args.amp:
        # test use mean reduction for everything
        scaler_g = GradScaler(init_scale=2.0**8, growth_factor=1.5)
        scaler_d = GradScaler(init_scale=2.0**8, growth_factor=1.5)

    start_step = resume_from_latest(
        autoencoder,
        discriminator,
        optimizer_g, 
        optimizer_d,
        scheduler_g,
        scheduler_d,
        scaler_g,
        scaler_d,
        output_dir, device) if args.resume else 0

    param_counts = count_parameters(autoencoder)
    print(f"### autoencoder's Total parameters: {param_counts['total']:,}")
    print(f"### autoencoder's Trainable parameters: {param_counts['trainable']:,}")
    print(f"### autoencoder's Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"### autoencoder's Parameters by layer type: {param_counts['by_layer_type']}")

    param_counts = count_parameters(discriminator)
    print(f"### discriminator's Total parameters: {param_counts['total']:,}")
    print(f"### discriminator's Trainable parameters: {param_counts['trainable']:,}")
    print(f"### discriminator's Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"### discriminator's Parameters by layer type: {param_counts['by_layer_type']}")

    # progress_bar = tqdm(
    #     range(0, args.max_train_steps),
    #     initial=start_step,
    #     desc="Steps",
    # )
    # progress_bar.set_description("Steps")

    

    # Training
    # Initialize variables
    # val_interval = args.val_interval
    best_val_recon_epoch_loss = 10000000.0
    # total_step = 0
    # start_epoch = 0
    # max_epochs = args.n_epochs

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite_loader(dataloader_train)

    # Setup validation inferer
    val_inferer = (
        SlidingWindowInferer(
            roi_size=args.val_sliding_window_patch_size,
            sw_batch_size=1,
            progress=False,
            overlap=0.0,
            device=torch.device("cpu"),
            sw_device=device,
        )
        if args.val_sliding_window_patch_size
        else SimpleInferer()
    )


    def loss_weighted_sum(losses):
        return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]


    # Training and validation loops
    with trange(start_step, 
                args.max_train_steps + 1, 
                desc="Training", 
                initial=start_step,
                total=args.max_train_steps + 1) as t:
        
        step_times = []
        for step in t:
            autoencoder.train()
            discriminator.train()
            
            step_start = time.time()
        
            batch = next(train_iter)
            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                # Train Generator
                reconstruction, z_mu, z_sigma = autoencoder(images)
                # print("images.shape:", images.shape, flush=True)
                # print("reconstruction.shape:", reconstruction.shape, flush=True)
                # print("z_mu.shape:", z_mu.shape, flush=True)
                losses = {
                    "recons_loss": intensity_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "p_loss": loss_perceptual(reconstruction.float(), images.float()),
                }
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

                if args.amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()
                scheduler_g.step()

                # Train Discriminator
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                if args.amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()
                scheduler_d.step()

            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            if step % 1000 == 0 and step > 0:
                avg_step = sum(step_times) / len(step_times)
                print(f"[Step {step}] Avg step time: {avg_step:.2f} sec, \
                        ETA: {(args.max_train_steps-step)*avg_step/60:.1f} min")
                
            # Log training loss
            t.set_postfix({'Total_g_loss': f"{loss_g.item():.4f}",
                           'Total_d_loss': f"{loss_d.item():.4f}",
                           'lr': f"{scheduler_g.get_lr()}",
                           'step_elapsed': f"{step_elapsed:.2f}sec"})
            
            if args.report_to:
                for loss_name, loss_value in losses.items():
                    # tensorboard_writer.add_scalar(f"train_{loss_name}_iter", loss_value.item(), total_step)
                    wandb.log({f"train_g/train_{loss_name}_iter": loss_value.item()}, step=step)
                    # train_epoch_losses[loss_name] += loss_value.item()
                # tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
                # tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
                # tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)
                wandb.log({"train_d/train_adv_loss_iter": generator_loss.item()}, step=step)
                wandb.log({"train_d/train_fake_loss_iter": loss_d_fake.item()}, step=step)
                wandb.log({"train_d/train_real_loss_iter": loss_d_real.item()}, step=step)

            if step % args.checkpointing_steps == 0 and step > start_step:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"autoencoder": autoencoder.state_dict(), 
                            "discriminator": discriminator.state_dict(), 
                            "optimizer_g": optimizer_g.state_dict(), 
                            "optimizer_d": optimizer_d.state_dict(), 
                            "scheduler_g": scheduler_g.state_dict(), 
                            "scheduler_d": scheduler_d.state_dict(), 
                            "scaler_g": scaler_g.state_dict(), 
                            "scaler_d": scaler_d.state_dict(), 
                            "step": step}, 
                           os.path.join(ckpt_dir, "model.pt"))
                print("Save trained autoencoder.", flush=True)
                # print("Save trained discriminator to", trained_d_path)

                # for key in train_epoch_losses:
                #     train_epoch_losses[key] /= len(dataloader_train)
                # print(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")
                # for loss_name, loss_value in train_epoch_losses.items():
                #     tensorboard_writer.add_scalar(f"train_{loss_name}_epoch", loss_value, epoch)
                # torch.save(autoencoder.state_dict(), trained_g_path)
                # torch.save(discriminator.state_dict(), trained_d_path)

            # Validation
            if step % args.validation_steps == 0 and step > start_step:
                autoencoder.eval()
                val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
                # val_loader_iter = iter(dataloader_val)
                for i, batch in enumerate(dataloader_val):
                    if i == 0: # too much time spent on valid stage
                        with torch.no_grad():
                            with autocast("cuda", enabled=args.amp):
                                images = batch["image"]
                                # images = torch.as_tensor(images) ###
                                reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images)
                                # reconstruction, _, _ = val_inferer(network=autoencoder, inputs=images)
                                reconstruction = reconstruction.to(device)
                                val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images.to(device)).item()
                                val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                                val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images.to(device)).item()

                # for key in val_epoch_losses:
                #     val_epoch_losses[key] /= len(dataloader_val)

                val_loss_g = loss_weighted_sum(val_epoch_losses)
                print(f"Step {step} val_vae_loss {val_loss_g}: {val_epoch_losses}.")

                if val_loss_g < best_val_recon_epoch_loss:
                    best_val_recon_epoch_loss = val_loss_g
                    # trained_g_path_epoch = f"{trained_g_path[:-3]}_epoch{epoch}.pt"
                    # torch.save(autoencoder.state_dict(), trained_g_path_epoch)
                    print(f"Got best val vae loss at step {step}.", flush=True)
                    # print("Save trained autoencoder to", trained_g_path_epoch)

                if args.report_to:
                    for loss_name, loss_value in val_epoch_losses.items():
                        # tensorboard_writer.add_scalar(loss_name, loss_value, epoch)
                        wandb.log({f"valid/{loss_name}": loss_value}, step=step)

                # Monitor scale_factor
                # We'd like to tune kl_weights in order to make scale_factor close to 1.
                scale_factor_sample = 1.0 / z_mu.flatten().std()
                if args.report_to:
                    # tensorboard_writer.add_scalar("val_one_sample_scale_factor", scale_factor_sample, epoch)
                    wandb.log({"valid/val_one_sample_scale_factor": scale_factor_sample}, step=step)

                # Monitor reconstruction result
                center_loc_axis = find_label_center_loc(images[0, 0, ...])
                # vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
                # vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)
                vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
                vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())
                vis_image = (vis_image * 255).astype(np.uint8)
                vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)
                vis_recon_image = (vis_recon_image - vis_recon_image.min()) / (vis_recon_image.max() - vis_recon_image.min())
                vis_recon_image = (vis_recon_image * 255).astype(np.uint8)

                if args.report_to:
                    # tensorboard_writer.add_image(
                    #     "val_orig_img",
                    #     vis_image.transpose([2, 0, 1]),
                    #     epoch,
                    # )
                    # tensorboard_writer.add_image(
                    #     "val_recon_img",
                    #     vis_recon_image.transpose([2, 0, 1]),
                    #     epoch,
                    # )
                    wandb.log({"val_orig_img": wandb.Image(vis_image)}, step=step)
                    wandb.log({"val_recon_img": wandb.Image(vis_recon_image)}, step=step)

                # show_image(vis_image, title="val image")
                # show_image(vis_recon_image, title="val recon result")

if __name__ == '__main__':
    try:
        main()
    finally:
        if wandb.run is not None:
            wandb.finish()
