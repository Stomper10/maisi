from __future__ import annotations
import os
import json
import wandb
import logging
import argparse
from pathlib import Path
from datetime import datetime
import signal  # [추가]
import sys     # [추가]

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers.rectified_flow import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.utils import define_instance
from utils import count_parameters

# [추가] Graceful Shutdown을 위한 전역 변수 및 핸들러
SHUTDOWN_REQUESTED = False

def graceful_shutdown_handler(signum, frame):
    """SIGTERM 신호를 받으면 SHUTDOWN_REQUESTED 플래그를 True로 설정합니다."""
    global SHUTDOWN_REQUESTED
    try:
        logging.getLogger("training").info(f"Received signal {signum}. Requesting graceful shutdown...")
    except Exception:
        print(f"Received signal {signum}. Requesting graceful shutdown...")
    SHUTDOWN_REQUESTED = True

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
    filenames = json_data[type]
    return [_item["image"].replace(".nii.gz", "_emb.nii.gz").split("/")[-1] for _item in filenames] ###

def prepare_data(
    data_files: list,
    device: torch.device,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
    shuffle_data: bool = True,
) -> DataLoader:
    """
    Prepare training data.

    Args:
        data_files (list): List of files.
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

    data_transforms_list = [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
            #monai.transforms.Lambdad(keys="cond", func=lambda x: _load_data_from_file(x, "cond")),
    ]
    if include_body_region:
        data_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index", func=lambda x: _load_data_from_file(x, "top_region_index")
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: _load_data_from_file(x, "bottom_region_index")
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]
    data_transforms = Compose(data_transforms_list)

    data_ds = monai.data.CacheDataset(
        data=data_files, transform=data_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return DataLoader(data_ds, num_workers=4, batch_size=batch_size, shuffle=shuffle_data)

def load_unet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if dist.is_initialized():
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet

def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor

def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)

def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

def train_one_epoch(
    epoch: int,
    step: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    ckpt_path: str, # [수정] 체크포인트 경로를 직접 받음
    amp: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        amp (bool): Use automatic mixed precision training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        step += 1
        images = train_data["image"].to(device)
        images = images * scale_factor

        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
        # We trained with only CT in this version
        if include_modality:
            # modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
            class_tensor = torch.tensor(train_data["modality_class"], dtype=torch.long).to(device)
        spacing_tensor = train_data["spacing"].to(device)
        print("### images.shape:", images.shape, flush=True) ###
        print('### train_data["modality_class"]:', train_data["modality_class"], flush=True)
        print('### train_data["modality_class"].shape:', train_data["modality_class"].unsqueeze(1).shape, flush=True) ###
        # context = train_data["modality_class"].unsqueeze(1).to(device) ###

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp): # with autocast(enabled=amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            #print(noisy_latent.shape, flush=True)
            #print(spacing_tensor, flush=True)

            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
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
            if include_modality: # t1n, t1c, t2w, t2f (prev was MRI vs CT)
                unet_inputs.update(
                    {
                        "class_labels": class_tensor
                    }
                )
            model_output = unet(**unet_inputs)

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                # predict noise
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                # predict sample
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                # predict velocity
                model_gt = images - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            loss = loss_pt(model_output.float(), model_gt.float())

            # if amp:  # Gradient scaling should be inside autocast
            #     scaler.scale(loss).backward()
            # else:
            #     loss.backward()

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            wandb.log({f"train/loss": loss.item()}, step=step)
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )

        if SHUTDOWN_REQUESTED:
            if local_rank == 0:
                logger.info("Shutdown requested. Saving final checkpoint...")
                save_checkpoint(
                    epoch=epoch,
                    step=step,
                    unet=unet,
                    loss_torch_epoch=loss.item(),
                    val_loss_epoch=float('inf'),
                    num_train_timesteps=num_train_timesteps,
                    scale_factor=scale_factor,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    ckpt_path=ckpt_path,
                )
                logger.info(f"Final checkpoint saved to {ckpt_path}. Exiting gracefully.")
            if dist.is_initialized():
                dist.barrier()
            sys.exit(0)

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch, step

def validate_one_epoch(
    unet: torch.nn.Module,
    val_loader: DataLoader,
    loss_pt: torch.nn.L1Loss,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_train_timesteps: int,
    device: torch.device,
    amp: bool = True,
) -> torch.Tensor:
    """
    Validate the model for one epoch.
    """
    unet.eval()
    val_loss = torch.zeros(2, dtype=torch.float, device=device)
    unet_module = unet.module if dist.is_initialized() else unet
    include_body_region = unet_module.include_top_region_index_input
    include_modality = unet_module.num_class_embeds is not None

    with torch.no_grad():
        for val_data in val_loader:
            images = val_data["image"].to(device)
            images = images * scale_factor
            spacing_tensor = val_data["spacing"].to(device)

            with autocast(enabled=amp):
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

                unet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "spacing_tensor": spacing_tensor,
                }

                if include_body_region:
                    top_region_index_tensor = val_data["top_region_index"].to(device)
                    bottom_region_index_tensor = val_data["bottom_region_index"].to(device)
                    unet_inputs.update({
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    })

                if include_modality:
                    # modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
                    class_tensor = torch.tensor(val_data["modality_class"], dtype=torch.long).to(device)
                    unet_inputs.update({"class_labels": class_tensor})

                model_output = unet(**unet_inputs)

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    model_gt = images
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    model_gt = images - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from "
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]"
                    )

                loss = loss_pt(model_output.float(), model_gt.float())

            val_loss[0] += loss.item()
            val_loss[1] += 1.0

    if dist.is_initialized():
        dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)

    return val_loss



def save_checkpoint(
    epoch: int,
    step: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    val_loss_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    optimizer: torch.optim.Optimizer, ###
    lr_scheduler: torch.optim.lr_scheduler, ###
    scaler: GradScaler, ###
    ckpt_path: str,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "loss": loss_torch_epoch,
            "val_loss": val_loss_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
            "optimizer": optimizer.state_dict(), ###
            "lr_scheduler": lr_scheduler.state_dict(), ###
            "scaler": scaler.state_dict(), ###
        },
        ckpt_path,
    )


def diff_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, amp: bool = True, resume: bool = True, ###
) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use for training.
        amp (bool): Use automatic mixed precision training.
    """
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] val_data_list -> {args.val_json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")
        
        wandb.init(project="maisi_unet_brats", config=args)
        wandb.run.define_metric("step")
        wandb.run.define_metric("train/loss", step_metric="step")
        wandb.run.define_metric("valid/loss", step_metric="step")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    unet = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # trianing data
    filenames_train = load_filenames(args.json_data_list, "training")
    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")

    train_files = []
    for _i in range(len(filenames_train)):
        str_img = os.path.join(args.embedding_base_dir, filenames_train[_i])
        if not os.path.exists(str_img):
            continue

        str_info = os.path.join(args.embedding_base_dir, filenames_train[_i]) + ".json"
        train_files_i = {"image": str_img, "spacing": str_info} ###
        #train_files_i["modality_class"] = filenames_train[_i]
        if include_body_region:
            train_files_i["top_region_index"] = str_info
            train_files_i["bottom_region_index"] = str_info
        if include_modality:
            train_files_i["modality_class"] = str_info
        train_files.append(train_files_i)

    # validation data
    filenames_valid = load_filenames(args.val_json_data_list, "validation")
    if local_rank == 0:
        logger.info(f"num_files_valid: {len(filenames_valid)}")

    valid_files = []
    for _i in range(len(filenames_valid)):
        str_img = os.path.join(args.embedding_base_dir, filenames_valid[_i])
        if not os.path.exists(str_img):
            continue

        str_info = os.path.join(args.embedding_base_dir, filenames_valid[_i]) + ".json"
        valid_files_i = {"image": str_img, "spacing": str_info} ###
        if include_body_region:
            valid_files_i["top_region_index"] = str_info
            valid_files_i["bottom_region_index"] = str_info
        if include_modality:
            valid_files_i["modality_class"] = str_info
        valid_files.append(valid_files_i)

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]
        valid_files = partition_dataset(
            data=valid_files, shuffle=False, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]

    train_loader = prepare_data(
        train_files, 
        device, 
        args.diffusion_unet_train["cache_rate"], 
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
        shuffle_data=True,
    )
    valid_loader = prepare_data(
        valid_files, 
        device, 
        args.diffusion_unet_train["cache_rate"], 
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
        shuffle_data=False,
    )

    # count params
    param_counts = count_parameters(unet)
    print(f"### UNET's Total parameters: {param_counts['total']:,}", flush=True)
    print(f"### UNET's Trainable parameters: {param_counts['trainable']:,}", flush=True)
    print(f"### UNET's Non-trainable parameters: {param_counts['non_trainable']:,}", flush=True)
    print(f"### UNET's Parameters by layer type: {param_counts['by_layer_type']}", flush=True)

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train[
        "batch_size"
    ]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()

    scaler = None
    if args.amp:
        scaler = GradScaler() ### scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")
    early_stop_patience = getattr(args, "early_stop_patience", 5)
    early_stop_min_delta = getattr(args, "early_stop_min_delta", 0.0)
    no_improve = 0

    if resume:
        start_epoch, best_val_loss, step = resume_from_latest(
            unet,
            optimizer, 
            lr_scheduler,
            scaler,
            args.model_dir, 
            device)
    else:
        start_epoch, best_val_loss, step = 0, float("inf"), 0

    for epoch in range(start_epoch, args.diffusion_unet_train["n_epochs"]): ###
        latest_ckpt_path = os.path.join(args.model_dir, args.model_filename)
        loss_torch, step = train_one_epoch(
            epoch=epoch,
            step=step,
            unet=unet,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_pt=loss_pt,
            scaler=scaler,
            scale_factor=scale_factor,
            noise_scheduler=noise_scheduler,
            num_images_per_batch=args.diffusion_unet_train["batch_size"],
            num_train_timesteps=args.noise_scheduler["num_train_timesteps"],
            device=device,
            logger=logger,
            local_rank=local_rank,
            ckpt_path=latest_ckpt_path, # [수정] ckpt_path 전달
            amp=amp,
        )
        loss_torch = loss_torch.tolist()

        # Validation step
        val_loss_torch = validate_one_epoch(
            unet, valid_loader, loss_pt, scale_factor, noise_scheduler,
            args.noise_scheduler["num_train_timesteps"], device, amp=amp
        )
        val_loss_torch = val_loss_torch.tolist()

        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            val_loss_epoch = val_loss_torch[0] / val_loss_torch[1]
            wandb.log({f"valid/loss": val_loss_epoch}, step=step)
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")
            logger.info(f"Epoch {epoch + 1} average validation loss: {val_loss_epoch:.4f}")

            latest_ckpt_path = os.path.join(args.model_dir, args.model_filename)
            save_checkpoint(
                epoch,
                step,
                unet,
                loss_torch_epoch,
                val_loss_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                optimizer, ###
                lr_scheduler, ###
                scaler, ###
                latest_ckpt_path
            )

            # Save the best model checkpoint based on validation loss
            if val_loss_epoch < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss_epoch
                no_improve = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving best model checkpoint.")
                best_ckpt_path = os.path.join(args.model_dir, args.model_filename.replace('.pt', '_best.pt'))
                save_checkpoint(
                    epoch, step, unet, loss_torch_epoch, val_loss_epoch, args.noise_scheduler["num_train_timesteps"],
                    scale_factor, optimizer, lr_scheduler, scaler, best_ckpt_path
                )
            else:
                no_improve += 1
                logger.info(f"[EarlyStopping] No improve ({no_improve}/{early_stop_patience})")

            if no_improve >= early_stop_patience:
                logger.info("[EarlyStopping] Patience exceeded. Stopping training.")
                break

    if dist.is_initialized():
        dist.destroy_process_group()

def resume_from_latest(unet,
                       optimizer, 
                       lr_scheduler,
                       scaler,
                       model_dir, device):
    if not os.path.exists(model_dir):
        print("[Resume] No checkpoint directory found. Starting fresh.")
        return 0, float("inf"), 0

    ckpts = [d for d in os.listdir(model_dir)]
    if len(ckpts) == 0:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0, float("inf"), 0

    # ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    # latest_ckpt_dir = ckpts[-1]
    # print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")
    
    path = os.path.join(model_dir, "diff_unet_ckpt.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    unet.load_state_dict(ckpt["unet_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    #scale_factor = ckpt["scale_factor"]

    return ckpt.get("epoch", 0) + 1, ckpt.get("val_loss", float("inf")), ckpt.get("step", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")
    parser.add_argument("--resume", action="store_true") ###

    args = parser.parse_args()
    diff_model_train(args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp, args.resume) ###
