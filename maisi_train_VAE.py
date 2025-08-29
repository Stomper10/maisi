import os
import sys
import json
import yaml
import wandb
import signal
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.config import print_config
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.data import CacheDataset, DataLoader, DistributedSampler

from utils import count_parameters
from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot

warnings.filterwarnings("ignore")
print_config()

def setup_ddp(): # DDP CHANGE: 분산 환경 초기화 함수
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp(): # DDP CHANGE: 분산 환경 정리 함수
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

SHUTDOWN_REQUESTED = False
def graceful_shutdown(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"\n[!] Received signal {signum}. Requesting graceful shutdown...")
    SHUTDOWN_REQUESTED = True
signal.signal(signal.SIGTERM, graceful_shutdown)

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
    parser.add_argument("--model_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_vae_train.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cpus_per_task", type=int, default=8, help="Number of CPUs allocated per task by Slurm.")
    args = parser.parse_args()
    args.run_name = get_run_name(args.run_name)
    config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)
    config_train_dict = json.load(open(args.train_config_path, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
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
        return 0, float("inf")

    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, d))
        and os.path.exists(os.path.join(output_dir, d, "model.pt"))
    ]
    if len(ckpts) == 0:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0, float("inf")

    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    latest_ckpt_dir = ckpts[-1]
    print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")
    
    path = os.path.join(output_dir, latest_ckpt_dir, "model.pt")
    ckpt = torch.load(path, map_location=device)
    
    autoencoder.load_state_dict(ckpt["autoencoder"])
    discriminator.load_state_dict(ckpt["discriminator"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])
    if "scheduler_g" in ckpt: scheduler_g.load_state_dict(ckpt["scheduler_g"])
    if "scheduler_d" in ckpt: scheduler_d.load_state_dict(ckpt["scheduler_d"])
    if "scaler_g" in ckpt and scaler_g is not None: scaler_g.load_state_dict(ckpt["scaler_g"])
    if "scaler_d" in ckpt and scaler_d is not None: scaler_d.load_state_dict(ckpt["scaler_d"])

    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf"))

def prepare_image_for_logging(image_tensor, center_loc):
    """3D 텐서를 wandb에 로깅할 수 있는 2D 이미지로 변환합니다."""
    image_tensor_cpu = image_tensor.cpu()
    vis_img_np = get_xyz_plot(image_tensor_cpu, center_loc, mask_bool=False)
    min_val, max_val = vis_img_np.min(), vis_img_np.max()
    if max_val - min_val > 1e-6:
        vis_img_np = (vis_img_np - min_val) / (max_val - min_val)
    else:
        vis_img_np = np.zeros_like(vis_img_np)
    vis_img_uint8 = (vis_img_np * 255).astype(np.uint8)
    return wandb.Image(vis_img_uint8)

def main():
    setup_ddp() # DDP CHANGE: 분산 설정 초기화
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    args = load_config()
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = torch.float16 if args.weight_dtype == "fp16" else torch.float32

    # OPTIMIZATION: 각 프로세스에 다른 시드를 주어 데이터 증강의 다양성 확보
    set_determinism(seed=args.seed + rank)
    # OPTIMIZATION: cudnn.benchmark는 입력 크기가 일정할 때 가장 효과적이므로 활성화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if rank == 0:
        run_name = args.run_name
        print("[Config] Loaded hyperparameters:")
        print(yaml.dump(vars(args), sort_keys=False))
        if args.report_to:
            wandb.init(project="maisi_ex", config=args, name=run_name)
            wandb.run.define_metric("step")
            wandb.run.define_metric("train_ae/loss", step_metric="step")
            wandb.run.define_metric("valid_ae/loss", step_metric="step")
            wandb.run.define_metric("train_ae/lr", step_metric="step")
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    # Data
    train_label_df = pd.read_csv(args.train_label_dir)
    valid_label_df = pd.read_csv(args.valid_label_dir)[:args.num_valid] ### 1000
    train_files_list = [{"image": os.path.join(args.data_dir, image_name)} for image_name in train_label_df["rel_path"]]
    val_files_list = [{"image": os.path.join(args.data_dir, image_name)} for image_name in valid_label_df["rel_path"]]

    def add_assigned_class_to_datalist(datalist, classname): # 데이터에 'class' 키를 추가하는 함수
        for item in datalist:
            item["class"] = classname
        return datalist

    train_files = add_assigned_class_to_datalist(train_files_list, "mri")
    val_files = add_assigned_class_to_datalist(val_files_list, "mri")

    train_transform = VAE_Transform(
        is_train=True, 
        random_aug=args.random_aug, 
        k=4, 
        patch_size=args.patch_size, 
        output_dtype=weight_dtype, 
        spacing_type=args.spacing_type, 
        image_keys=["image"]
    )
    val_transform = VAE_Transform(
        is_train=False, 
        random_aug=False, 
        k=4, 
        output_dtype=weight_dtype, 
        image_keys=["image"]
    )

    # Build dataloader
    if rank == 0: print(f"Total number of training data is {len(train_files)}.")
    train_dataset = CacheDataset(data=train_files, transform=train_transform, cache_rate=args.cache, num_workers=args.cpus_per_task // world_size)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    workers_per_gpu = args.cpus_per_task // world_size # 각 프로세스가 사용할 수 있는 CPU 코어 수에 맞춰 num_workers 설정
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=workers_per_gpu, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    if rank == 0: # DDP CHANGE: Validation은 rank 0에서만 진행하므로, sampler 불필요
        print(f"Total number of validation data is {len(val_files)}.")
        dataset_val = CacheDataset(data=val_files, transform=val_transform, cache_rate=args.cache, num_workers=workers_per_gpu)
        dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=workers_per_gpu, shuffle=False, pin_memory=True)
        print("### Train Transform ###")
        for i, t in enumerate(train_transform.transform_dict["mri"].transforms):
            print(f"[{i}] {t}")
        print("\n### Validation Transform ###")
        for i, t in enumerate(val_transform.transform_dict["mri"].transforms):
            print(f"[{i}] {t}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # if rank == 0:
    #     print("Instantiating SafeAutoencoderKL model.")
    # ae_params = args.autoencoder_def
    # ae_params.pop('_target_')
    # autoencoder = SafeAutoencoderKL(**ae_params).to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    if rank == 0: print("Warming up models for torch.compile() with AMP...")
    dummy_input = torch.randn(1, 1, *args.patch_size, device=device, dtype=weight_dtype)
    with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        autoencoder(dummy_input)
        discriminator(dummy_input)

    # OPTIMIZATION: torch.compile로 모델 컴파일하여 연산 가속화 (PyTorch 2.0+ 필수)
    if rank == 0: print("Compiling models with torch.compile()...")
    autoencoder = torch.compile(autoencoder)
    discriminator = torch.compile(discriminator)
    autoencoder = DDP(autoencoder, device_ids=[local_rank], find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=True)

    # Training config
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.max_train_steps)
    scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.max_train_steps)
    scaler_g, scaler_d = (GradScaler(), GradScaler()) if args.amp else (None, None)

    start_step, best_val_recon_epoch_loss = 0, float("inf")
    if args.resume:
        run_name_for_resume = args.run_name # 실제 실행시 run_name을 명시해야 함
        output_dir_for_resume = os.path.join(args.output_dir, run_name_for_resume)
        
        # DDP에서는 DDP로 감싸기 전에 모델 가중치를 로드해야 함. 현재 코드는 이 순서를 따르고 있음.
        # DDP 래핑 전에 호출해야 하므로, autoencoder.module이 아닌 autoencoder를 전달
        if rank == 0:
             start_step, best_val_recon_epoch_loss = resume_from_latest(
                autoencoder.module, discriminator.module, optimizer_g, optimizer_d,
                scheduler_g, scheduler_d, scaler_g, scaler_d, output_dir_for_resume, device)
        
        # DDP CHANGE: rank 0에서 읽은 step과 loss 값을 모든 프로세스에 동기화
        sync_data = torch.tensor([start_step, best_val_recon_epoch_loss], dtype=torch.float32, device=device)
        dist.broadcast(sync_data, src=0)
        start_step, best_val_recon_epoch_loss = int(sync_data[0].item()), sync_data[1].item()
        
    if rank == 0:
        param_counts = count_parameters(autoencoder.module)
        print(f"### autoencoder's Trainable parameters: {param_counts['trainable']:,}")
        param_counts = count_parameters(discriminator.module)
        print(f"### discriminator's Trainable parameters: {param_counts['trainable']:,}")

    def infinite_loader(loader, sampler):
        epoch = 0
        while True:
            sampler.set_epoch(epoch) # DDP CHANGE: 매 에포크마다 샘플러 시드 변경
            for batch in loader:
                yield batch
            epoch += 1
            
    train_iter = infinite_loader(dataloader_train, train_sampler)

    # --- 학습 루프 ---
    progress_bar = trange(start_step, args.max_train_steps + 1,
                          desc=f"Training on Rank {rank}",
                          initial=start_step, total=args.max_train_steps + 1,
                          disable=(rank != 0))
    
    for step in progress_bar:
        autoencoder.train()
        discriminator.train()
        
        batch = next(train_iter)
        # OPTIMIZATION: non_blocking=True로 데이터 전송과 연산 오버랩 시도
        images = batch["image"].to(device, non_blocking=True).contiguous()

        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            reconstruction, z_mu, z_sigma = autoencoder(images)
            z_sigma = torch.clamp(z_sigma, min=1e-6)
            losses = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": KL_loss(z_mu, z_sigma),
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + \
                     args.perceptual_weight * losses["p_loss"] + args.adv_weight * generator_loss
        
        if args.amp:
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            clip_grad_norm_(autoencoder.parameters(), 1.0)     # clipping
            scaler_g.step(optimizer_g)
            scaler_g.update()
        else:
            loss_g.backward()
            clip_grad_norm_(autoencoder.parameters(), 1.0)     # clipping
            optimizer_g.step()
        scheduler_g.step()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

        if args.amp:
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(optimizer_d)
            clip_grad_norm_(discriminator.parameters(), 1.0)     # clipping
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            loss_d.backward()
            clip_grad_norm_(discriminator.parameters(), 1.0)     # clipping
            optimizer_d.step()
        scheduler_d.step()

        # DDP CHANGE: 모든 GPU의 loss를 평균내어 로그 기록 (선택적이지만 정확성을 위해 권장)
        dist.all_reduce(loss_g, op=dist.ReduceOp.AVG)
        dist.all_reduce(loss_d, op=dist.ReduceOp.AVG)

        if rank == 0:
            progress_bar.set_postfix({'Total_g_loss': f"{loss_g.item():.4f}", 'Total_d_loss': f"{loss_d.item():.4f}"})
            if args.report_to and step % 100 == 0: # 로그 기록 빈도 조절
                log_data = {
                    "train/learning_rate": scheduler_g.get_last_lr()[0],
                    "train/loss_g_total": loss_g.item(),
                    "train/loss_d_total": loss_d.item(),
                }
                for loss_name, loss_value in losses.items():
                    log_data[f"train/generator/{loss_name}"] = loss_value.item()
                log_data["train/discriminator/adv_g_loss"] = generator_loss.item()
                log_data["train/discriminator/d_fake_loss"] = loss_d_fake.item()
                log_data["train/discriminator/d_real_loss"] = loss_d_real.item()
                wandb.log(log_data, step=step)

            # --- 체크포인트 저장 및 Validation ---
            is_time_to_save = (step % args.checkpointing_steps == 0 and step > start_step)
            if is_time_to_save or SHUTDOWN_REQUESTED:
                state = {
                    # DDP CHANGE: .module을 통해 원본 모델의 state_dict 저장
                    "autoencoder": autoencoder.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                    "step": step,
                    "best_val_loss": float(best_val_recon_epoch_loss),
                }
                if args.amp:
                    state["scaler_g"] = scaler_g.state_dict()
                    state["scaler_d"] = scaler_d.state_dict()
                
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(state, os.path.join(ckpt_dir, "model.pt"))
                print(f"\nSaved checkpoint to {ckpt_dir}", flush=True)
                
                if SHUTDOWN_REQUESTED:
                    print(f"Graceful shutdown: Final checkpoint saved to {ckpt_dir}. Exiting.", flush=True)
                    break # 루프 종료

            if step % args.validation_steps == 0 and step > start_step:
                autoencoder.eval()
                val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
                val_inferer = SlidingWindowInferer(
                    roi_size=args.val_sliding_window_patch_size, 
                    sw_batch_size=1, 
                    overlap=0.5, 
                    device=device, #torch.device("cpu"), 
                    sw_device=device
                )
                
                with torch.no_grad():
                    for val_batch in dataloader_val:
                        val_images = val_batch["image"]
                        # DDP CHANGE: .module을 통해 원본 모델로 추론
                        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                            reconstruction, z_mu_val, z_sigma_val = dynamic_infer(val_inferer, autoencoder.module, val_images)
                            z_sigma_val = torch.clamp(z_sigma_val, min=1e-6)
                        reconstruction = reconstruction.to(device)
                        val_images = val_images.to(device)
                        val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, val_images).item()
                        val_epoch_losses["kl_loss"] += KL_loss(z_mu_val, z_sigma_val).item()
                        val_epoch_losses["p_loss"] += loss_perceptual(reconstruction.float(), val_images.float()).item()
                
                for key in val_epoch_losses:
                    val_epoch_losses[key] /= len(dataloader_val)

                val_loss_g = val_epoch_losses["recons_loss"] + \
                             args.kl_weight * val_epoch_losses["kl_loss"] + \
                             args.perceptual_weight * val_epoch_losses["p_loss"]
                
                print(f"\nStep {step} Total Val Loss: {val_loss_g:.4f}, Details: {val_epoch_losses}")

                if args.report_to:
                    log_data = {
                        "valid/total_loss": val_loss_g,
                        "valid/recon_loss": val_epoch_losses["recons_loss"],
                        "valid/kl_loss": val_epoch_losses["kl_loss"],
                        "valid/p_loss": val_epoch_losses["p_loss"],
                        "valid/scale_factor": (1.0 / z_mu_val.flatten().std()).item()
                    }
                    center_loc = find_label_center_loc(val_images[0, 0, ...])
                    log_data["valid/original_image"] = prepare_image_for_logging(val_images[0], center_loc)
                    log_data["valid/reconstructed_image"] = prepare_image_for_logging(reconstruction[0], center_loc)
                    wandb.log(log_data, step=step)
                    
        dist.barrier() # wait for rank 0 valid process
        shutdown_tensor = torch.tensor([0], device=device)
        if rank == 0 and SHUTDOWN_REQUESTED:
            shutdown_tensor[0] = 1
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor[0] == 1:
            if rank == 0:
                print("Shutdown signal received and synced across all ranks. Exiting training loop gracefully.")
            break

    if SHUTDOWN_REQUESTED:
        sys.exit(0)
    
    # DDP CHANGE: 모든 프로세스가 끝날 때까지 대기 후 정리
    dist.barrier()
    cleanup_ddp()

if __name__ == '__main__':
    main()