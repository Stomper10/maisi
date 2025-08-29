import argparse
import json
import logging
import os
import sys
import signal
import wandb
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from scripts.utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from scripts.diff_model_setting import setup_logging
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

def save_checkpoint(
    epoch: int,
    step: int,
    controlnet: torch.nn.Module,
    loss_torch_epoch: float,
    val_loss_epoch: float,
    optimizer: torch.optim.Optimizer, ###
    lr_scheduler: torch.optim.lr_scheduler, ###
    scaler: GradScaler, ###
    ckpt_dir: str,
    #args: argparse.Namespace,
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
    ctrl_state_dict = controlnet.module.state_dict() if dist.is_initialized() else controlnet.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "loss": loss_torch_epoch,
            "val_loss": val_loss_epoch,
            "ctrl_state_dict": ctrl_state_dict,
            "optimizer": optimizer.state_dict(), ###
            "lr_scheduler": lr_scheduler.state_dict(), ###
            "scaler": scaler.state_dict(), ###
        },
        ckpt_dir,
    )

def resume_from_latest(controlnet,
                       optimizer, 
                       lr_scheduler,
                       scaler,
                       model_dir, device):
    if not os.path.exists(model_dir):
        print("[Resume] No checkpoint directory found. Starting fresh.")
        return 0, float("inf"), 0

    ckpts = [d for d in os.listdir(model_dir)]
    if len(ckpts) == 0 or "diff_ctrl_ckpt.pt" not in ckpts:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0, float("inf"), 0

    # ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    # latest_ckpt_dir = ckpts[-1]
    # print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")
    
    path = os.path.join(model_dir, "diff_ctrl_ckpt.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    controlnet.load_state_dict(ckpt["ctrl_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    #scale_factor = ckpt["scale_factor"]

    return ckpt.get("epoch", 0) + 1, ckpt.get("val_loss", float("inf")), ckpt.get("step", 0)

def validate_one_epoch(
    controlnet: torch.nn.Module,
    unet: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    noise_scheduler,
    scale_factor: float,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    """
    Validate the model for one epoch.
    """
    controlnet.eval()  # Set controlnet to evaluation mode
    # unet is already in eval mode from the main script

    # Determine model properties based on DDP status
    unet_module = unet.module if dist.is_initialized() else unet
    include_body_region = unet_module.include_top_region_index_input
    include_modality = unet_module.num_class_embeds is not None

    # Extract training parameters from args
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]

    val_epoch_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            if include_body_region:
                top_region_index_tensor = batch["top_region_index"].to(device)
                bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            if include_modality:
                # modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
                class_tensor = torch.tensor(batch["modality_class"], dtype=torch.long).to(device)
            spacing_tensor = batch["spacing"].to(device)

            with autocast("cuda", enabled=args.amp):
                noise_shape = list(images.shape)
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                controlnet_cond = binarize_labels(labels.as_tensor().to(torch.uint8)).float()

                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

                controlnet_inputs = {"x": noisy_latent, "timesteps": timesteps, "controlnet_cond": controlnet_cond}
                if include_modality:
                    controlnet_inputs.update({"class_labels": class_tensor})

                down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

                unet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "spacing_tensor": spacing_tensor,
                    "down_block_additional_residuals": down_block_res_samples,
                    "mid_block_additional_residual": mid_block_res_sample,
                }
                if include_body_region:
                    unet_inputs.update({
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    })
                if include_modality:
                    unet_inputs.update({"class_labels": class_tensor})

                model_output = unet(**unet_inputs)

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    model_gt = images
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    model_gt = images - noise
                else:
                    raise ValueError("Unsupported prediction type")

                if weighted_loss > 1.0:
                    weights = torch.ones_like(images).to(images.device)
                    interpolate_label = F.interpolate(labels, size=images.shape[2:], mode="nearest")
                    roi_mask = torch.zeros_like(interpolate_label, dtype=torch.bool)
                    for label in weighted_loss_label:
                        roi_mask = roi_mask | (interpolate_label == label)
                    weights[roi_mask.repeat(1, images.shape[1], 1, 1, 1)] = weighted_loss
                    loss = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
                else:
                    loss = F.l1_loss(model_output.float(), model_gt.float())

                val_epoch_loss += loss.item()

    # Average the loss over all validation batches
    avg_val_loss = torch.tensor(val_epoch_loss / (step + 1), device=device)
    return avg_val_loss


def main():
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "-e",
        "--env_config",
        default="./configs/environment_maisi_controlnet_train.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-t",
        "--model_config",
        default="./configs/config_maisi_controlnet_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument(
        "-c",
        "--model_def",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")
    parser.add_argument("--resume", action="store_true") ###
    args = parser.parse_args()

    # Step 0: configuration
    #logger = logging.getLogger("maisi.controlnet.training")
    logger = setup_logging("training")
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
    with open(args.model_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

        wandb.init(project="maisi_ctrl_brats", config=args)
        wandb.run.define_metric("step")
        wandb.run.define_metric("train/loss", step_metric="step")
        wandb.run.define_metric("valid/loss", step_metric="step")

    # Step 1: set data loader
    train_loader, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list_ctrl, ###
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
    )

    # Step 2: define diffusion model and controlnet
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
    # # load trained controlnet model if it is provided
    # if args.trained_controlnet_path is not None:
    #     if not os.path.exists(args.trained_controlnet_path):
    #         raise ValueError("Please download the trained ControlNet checkpoint.")
    #     controlnet.load_state_dict(
    #         torch.load(args.trained_controlnet_path, map_location=device, weights_only=False)["controlnet_state_dict"]
    #     )
    #     logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    # else:
    #     logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    noise_scheduler = define_instance(args, "noise_scheduler")

    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    param_counts = count_parameters(controlnet)
    print(f"### CTRL's Total parameters: {param_counts['total']:,}", flush=True)
    print(f"### CTRL's Trainable parameters: {param_counts['trainable']:,}", flush=True)
    print(f"### CTRL's Non-trainable parameters: {param_counts['non_trainable']:,}", flush=True)
    print(f"### CTRL's Parameters by layer type: {param_counts['by_layer_type']}", flush=True)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = None
    if args.amp:
        scaler = GradScaler("cuda") ### scaler = GradScaler("cuda")
    total_step = 0
    early_stop_patience = getattr(args, "early_stop_patience", 5)
    early_stop_min_delta = getattr(args, "early_stop_min_delta", 0.0)
    no_improve = 0

    if weighted_loss > 1.0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    if args.resume:
        start_epoch, best_loss, curr_step = resume_from_latest( ###
            controlnet,
            optimizer, 
            lr_scheduler,
            scaler,
            args.model_dir, device)
    else:
        start_epoch, best_loss, curr_step = 0, float("inf")
    
    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(start_epoch, n_epochs): ###
        
        # if rank == 0:
        #     current_lr = optimizer.param_groups[0]["lr"]
        #     logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")
        controlnet.train()
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            images = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            # get corresponding conditions
            if include_body_region:
                top_region_index_tensor = batch["top_region_index"].to(device)
                bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            # We trained with only CT in this version
            if include_modality:
                class_tensor = torch.tensor(batch["modality_class"], dtype=torch.long).to(device)
                # modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
            spacing_tensor = batch["spacing"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=args.amp): # with autocast("cuda", enabled=True):
                # generate random noise
                noise_shape = list(images.shape)
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                # use binary encoding to encode segmentation mask
                controlnet_cond = binarize_labels(labels.as_tensor().to(torch.uint8)).float()
                
                # create timesteps
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                # create noisy latent
                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

                # print("### cond.shape:", controlnet_cond.shape, flush=True)
                # print("### noisy_latent.shape:", noisy_latent.shape, flush=True)

                # get controlnet output
                # Create a dictionary to store the inputs
                controlnet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "controlnet_cond": controlnet_cond,
                }
                if include_modality:
                    controlnet_inputs.update(
                        {
                            "class_labels": class_tensor,
                        }
                    )
                down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

                # get diffusion network output
                # Create a dictionary to store the inputs
                unet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "spacing_tensor": spacing_tensor,
                    "down_block_additional_residuals": down_block_res_samples,
                    "mid_block_additional_residual": mid_block_res_sample,
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
                            "class_labels": class_tensor,
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

            if weighted_loss > 1.0:
                weights = torch.ones_like(images).to(images.device)
                roi = torch.zeros([noise_shape[0]] + [1] + noise_shape[2:]).to(images.device)
                interpolate_label = F.interpolate(labels, size=images.shape[2:], mode="nearest")
                # assign larger weights for ROI (tumor)
                for label in weighted_loss_label:
                    roi[interpolate_label == label] = 1
                weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = weighted_loss
                loss = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
            else:
                loss = F.l1_loss(model_output.float(), model_gt.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1
            curr_step += 1

            # [추가] 매 스텝마다 종료 신호 확인 및 체크포인트 저장
            if SHUTDOWN_REQUESTED:
                if rank == 0:
                    latest_ckpt_path = os.path.join(args.model_dir, "diff_ctrl_ckpt.pt")
                    logger.info("Shutdown requested. Saving final checkpoint...")
                    save_checkpoint(
                        epoch=epoch,
                        step=curr_step,
                        controlnet=controlnet,
                        loss_torch_epoch=loss.item(),
                        val_loss_epoch=best_loss, # 현재까지의 best_loss 저장
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        ckpt_dir=latest_ckpt_path,
                    )
                    logger.info(f"Final checkpoint saved to {latest_ckpt_path}. Exiting gracefully.")
                if use_ddp:
                    dist.barrier()
                sys.exit(0)

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step
                )
                wandb.log({f"train/loss": loss.item()}, step=curr_step)
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss.detach().cpu().item(),
                        time_left,
                    )
                )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        val_loss_epoch = validate_one_epoch(controlnet, unet, val_loader, noise_scheduler, scale_factor, device, args)

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)
            dist.all_reduce(val_loss_epoch, op=torch.distributed.ReduceOp.AVG)

        epoch_loss = epoch_loss.item()
        val_loss_epoch = val_loss_epoch.item()

        if rank == 0:
            wandb.log({f"valid/loss": val_loss_epoch}, step=curr_step)
            logger.info(f"Epoch {epoch + 1} average train loss: {epoch_loss:.4f}")
            logger.info(f"Epoch {epoch + 1} average validation loss: {val_loss_epoch:.4f}")

            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss, total_step)
            tensorboard_writer.add_scalar("val/val_controlnet_loss_epoch", val_loss_epoch, total_step) # <<< ADDED: Log validation loss
            # save controlnet only on master GPU (rank 0)
            # controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            # torch.save(
            #     {
            #         "epoch": epoch + 1,
            #         "loss": epoch_loss,
            #         "controlnet_state_dict": controlnet_state_dict,
            #     },
            #     f"{args.model_dir}/{args.exp_name}_current.pt",
            # )
            save_checkpoint(
                epoch,
                curr_step,
                controlnet,
                epoch_loss,
                val_loss_epoch,
                optimizer, ###
                lr_scheduler, ###
                scaler, ###
                os.path.join(args.model_dir, "diff_ctrl_ckpt.pt"),
                #args,
            )

            if val_loss_epoch < best_loss - early_stop_min_delta:
                best_loss = val_loss_epoch
                no_improve = 0
                logger.info(f"New best validation loss: {best_loss:.4f}. Saving best model.")
                save_checkpoint(
                    epoch,
                    curr_step,
                    controlnet,
                    epoch_loss,
                    val_loss_epoch,
                    optimizer, ###
                    lr_scheduler, ###
                    scaler, ###
                    os.path.join(args.model_dir, "diff_ctrl_ckpt_best.pt"),
                    #args,
                )
            else:
                no_improve += 1
                logger.info(f"[EarlyStopping] No improve ({no_improve}/{early_stop_patience})")

            if no_improve >= early_stop_patience:
                logger.info("[EarlyStopping] Patience exceeded. Stopping training.")
                break

        torch.cuda.empty_cache()
    if use_ddp:
        dist.destroy_process_group()



if __name__ == "__main__":
    main()