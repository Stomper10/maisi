from __future__ import annotations

import os
import sys
import json
import yaml
import wandb
import signal
import logging
import argparse
import warnings
from pathlib import Path
from tqdm import trange

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import monai
from monai.transforms import Compose
from monai.config import print_config
from monai.utils import first, set_determinism
from monai.data import CacheDataset, DataLoader, DistributedSampler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from utils import count_parameters
from scripts.utils import define_instance
from scripts.diff_model_setting import load_config, setup_logging
from wavelet_module import DWT3D
import bitsandbytes.optim as bnb_optim

warnings.filterwarnings("ignore")
if os.environ.get("RANK", "0") == "0":
    print_config()

def setup_ddp(): # DDP CHANGE: 분산 환경 초기화 함수
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp(): # DDP CHANGE: 분산 환경 정리 함수
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

SHUTDOWN_REQUESTED = False
def graceful_shutdown(signum, frame):
    global SHUTDOWN_REQUESTED
    logging.info(f"\n[!] Received signal {signum}. Requesting graceful shutdown...")
    SHUTDOWN_REQUESTED = True
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# --- FrequencyWeightedLoss 클래스 정의 --- ###
class FrequencyWeightedLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.dwt = DWT3D().to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_x0_voxel, target_x0_voxel):
        # 1. 공간 영역에서의 기본 L1 Loss 계산
        spatial_loss = self.l1_loss(pred_x0_voxel, target_x0_voxel)

        # 2. 웨이블릿 변환 적용 (B, C, D, H, W) -> (B, 8*C, D/2, H/2, W/2)
        pred_coeffs = self.dwt(pred_x0_voxel)
        target_coeffs = self.dwt(target_x0_voxel)
        
        # 채널(C) 차원을 기준으로 8개의 subband 분리
        pred_subbands = torch.chunk(pred_coeffs, 8, dim=1)
        target_subbands = torch.chunk(target_coeffs, 8, dim=1)

        # 3. 각 주파수 성분별 손실 계산 및 가중합
        low_wavelet_loss = self.l1_loss(pred_subbands[0], target_subbands[0]) # LLL subband
        high_wavelet_loss = 0
        for i in range(1, 8): # 7개의 High-frequency subbands
            high_wavelet_loss += self.l1_loss(pred_subbands[i], target_subbands[i])
        
        # 4. 최종 손실 결합
        return spatial_loss, low_wavelet_loss, high_wavelet_loss

def prepare_transform(include_body_region: bool = False):
    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    data_transforms_list = [
            monai.transforms.LoadImaged(keys=["latent"]), ###
            monai.transforms.EnsureChannelFirstd(keys=["latent"]), ###
            monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="cond", func=lambda x: _load_data_from_file(x, "cond")),
            monai.transforms.Lambdad(keys="cond", func=lambda x: torch.tensor([x[0]], dtype=torch.float32)), ### 👈
            # monai.transforms.Orientationd(keys=["image"], axcodes="RAS"), ###
            # monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False), ###
            # monai.transforms.Resized(keys=["image"], spatial_size=(128, 256, 128), mode="trilinear") ###
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

    return data_transforms # DataLoader(data_ds, num_workers=4, batch_size=batch_size, shuffle=shuffle_data)

def calculate_scale_factor(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    scale_factor_tensor = torch.zeros(1, device=device)
    if dist.get_rank() == 0:
        check_data = first(train_loader)
        z = check_data["latent"].to(device)
        scale_factor_tensor[0] = 1 / torch.std(z)
    dist.broadcast(scale_factor_tensor, src=0)
    logging.info(f"Synchronized scaling factor: {scale_factor_tensor.item():.4f}.")
    return scale_factor_tensor

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config_path", type=str, required=True)
    parser.add_argument("--train_config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--cpus_per_task", type=int, default=8, help="Number of CPUs allocated per task by Slurm.")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    env_config_dict = json.load(open(args.env_config_path, "r"))
    for k, v in env_config_dict.items():
        setattr(args, k, v)
    model_config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in model_config_dict.items():
        setattr(args, k, v)
    train_config_dict = json.load(open(args.train_config_path, "r"))
    for k, v in train_config_dict["diffusion_unet_inference"].items():
        setattr(args, k, v)
    for k, v in train_config_dict["diffusion_unet_train"].items():
        setattr(args, k, v)
    return args

def resume_from_latest(unet, optimizer, lr_scheduler, scaler, output_dir, device):
    if not os.path.exists(output_dir):
        print("[Resume] No checkpoint directory found. Starting fresh.")
        return 0, float("inf"), 0
    
    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, d))
        and os.path.exists(os.path.join(output_dir, d, "diff_unet_ckpt.pt"))
    ]
    if len(ckpts) == 0:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0, float("inf"), 0
    
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    latest_ckpt_dir = ckpts[-1]
    print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")

    path = os.path.join(output_dir, latest_ckpt_dir, "diff_unet_ckpt.pt")
    ckpt = torch.load(path, map_location=device)

    unet.load_state_dict(ckpt["unet_state_dict"])
    # optimizer.load_state_dict(ckpt["optimizer"]) #### deactivate when submitting fine-tuning job
    # if "lr_scheduler" in ckpt: lr_scheduler.load_state_dict(ckpt["lr_scheduler"]) #### deactivate when submitting fine-tuning job
    # if "scaler" in ckpt and scaler is not None: scaler.load_state_dict(ckpt["scaler"]) #### deactivate when submitting fine-tuning job

    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf")), ckpt.get("epoch", 0)

def reduce_mean_scalar(x):
    t = x.detach().reshape(1).to(torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

def main():
    setup_ddp() # DDP CHANGE: 분산 설정 초기화
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    logger = setup_logging("training")

    if rank == 0:
        logger.info("="*50)
        logger.info(f"✅ Training started with a total of {world_size} GPUs across all nodes.")
        logger.info("="*50)

    args = load_config()
    args.model_dir = args.model_dir + "_hwl" ####
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Using {device} of {world_size}")
    #weight_dtype = torch.float16 if args.weight_dtype == "fp16" else torch.float32

    if rank == 0:
        logger.info(f"[Opt] Using gradient accumulation with {args.gradient_accumulation_steps} steps.")

    # OPTIMIZATION: 각 프로세스에 다른 시드를 주어 데이터 증강의 다양성 확보
    set_determinism(seed=args.seed + rank)
    # OPTIMIZATION: cudnn.benchmark는 입력 크기가 일정할 때 가장 효과적이므로 활성화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    logger = setup_logging("training")
    logger.info(f"Using {device} of {world_size}")

    if rank == 0:
        logger.info("[Config] Loaded hyperparameters:")
        logger.info(yaml.dump(vars(args), sort_keys=False))
        logger.info(f"Training started on {world_size} GPUs.")
        if args.report_to:
            wandb.init(project="MAISI_UNET_UKB", config=args, name=args.run_name)
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # --- VAE 모델 로드 (HWL-Loss 계산에 필요) --- ###
    autoencoder = define_instance(args, "autoencoder_def")
    path = args.trained_autoencoder_path
    ckpt_vae = torch.load(path, map_location="cpu")
    autoencoder.load_state_dict(ckpt_vae["autoencoder"])
    autoencoder.decoder = autoencoder.decoder.to("cpu").float()
    autoencoder.post_quant_conv = autoencoder.post_quant_conv.to("cpu").float()
    autoencoder.encoder = None
    autoencoder.quant_conv_mu = None
    autoencoder.quant_conv_log_sigma = None
    autoencoder.eval() # VAE는 학습하지 않으므로 eval 모드로 설정
    
    # 7. VAE 가중치 동결 (이미 .half()로 보낸 파라미터만 동결해도 충분)
    for param in autoencoder.decoder.parameters():
        param.requires_grad = False
    for param in autoencoder.post_quant_conv.parameters():
        param.requires_grad = False
    if rank == 0: 
        logger.info(f"VAE weights loaded from {path}.")
        logger.info("VAE Decoder and post_quant_conv loaded onto CPU in fp32 and frozen.")

    voxel_transform = Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        #monai.transforms.CastToTyped(keys=["image"], dtype=torch.float16),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
        monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
        monai.transforms.Resized(keys=["image"], spatial_size=(128, 256, 128), mode="trilinear")
    ])

    def custom_collate(batch_list):
        # batch_list는 __getitem__에서 반환된 딕셔너리 B 개(배치 크기)를 담은 '리스트'입니다.
        # 1. Voxel 이미지를 제외한 나머지 데이터를 기본 collate로 묶습니다.
        # (주의: 'image' 키는 문자열(경로)이라 기본 collate가 실패할 수 있으니 수동으로 합니다.)
        collated = {}
        keys_to_stack = ["latent", "spacing", "cond"] # 텐서로 묶을 키
        if include_body_region: # config에 따라 동적으로 추가
            keys_to_stack.extend(["top_region_index", "bottom_region_index"])

        for key in keys_to_stack:
            collated[key] = torch.stack([item[key] for item in batch_list])

        # 2. Voxel 이미지는 "배치당 하나만" 무작위로 선택하여 로드합니다.
        batch_size = len(batch_list)
        idx_to_load = torch.randint(0, batch_size, (1,)).item()

        # 'image' 키에는 경로(string)가 들어 있습니다.
        image_path_to_load = batch_list[idx_to_load]["image"] 

        # *여기서* I/O와 변환이 발생합니다 (워커 프로세스에서 병렬로!)
        loaded_voxel_image = voxel_transform({"image": image_path_to_load})["image"]

        collated["target_voxel_image"] = loaded_voxel_image # (C, D, H, W)
        collated["target_voxel_idx"] = idx_to_load # 이 이미지가 원래 몇 번째 샘플인지 기록

        return collated

    unet = define_instance(args, "diffusion_unet_def").to(device)
    noise_scheduler = define_instance(args, "noise_scheduler")
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    num_train_timesteps = args.noise_scheduler['num_train_timesteps']
    # args.max_train_steps = (args.n_epochs * len(filenames_train)) / args.batch_size

    # trianing data
    with open(args.json_data_list, "r") as file:
        json_data_train = json.load(file)
    filenames_train = json_data_train["training"]
    train_files = []
    for _i in range(len(filenames_train)):
        str_lat = os.path.join(args.embedding_base_dir, filenames_train[_i]["image"].split("/")[4][:7] + "_emb.nii.gz")
        if not os.path.exists(str_lat):
            continue
        str_info = str_lat + ".json"
        train_files_i = {"latent": str_lat, "image": filenames_train[_i]["image"], "spacing": str_info, "cond": str_info} ###
        if include_body_region:
            train_files_i["top_region_index"] = str_info
            train_files_i["bottom_region_index"] = str_info
        train_files.append(train_files_i)

    # validation data
    with open(args.val_json_data_list, "r") as file:
        json_data_valid = json.load(file)
    filenames_valid = json_data_valid["validation"][:args.num_valid]
    valid_files = []
    for _i in range(len(filenames_valid)):
        str_lat = os.path.join(args.embedding_base_dir, filenames_valid[_i]["image"].split("/")[4][:7] + "_emb.nii.gz")
        if not os.path.exists(str_lat):
            continue
        str_info = str_lat + ".json"
        valid_files_i = {"latent": str_lat, "image": filenames_valid[_i]["image"], "spacing": str_info, "cond": str_info} ###
        if include_body_region:
            valid_files_i["top_region_index"] = str_info
            valid_files_i["bottom_region_index"] = str_info
        valid_files.append(valid_files_i)

    if rank == 0:
        logger.info(f"Total number of training data is {len(train_files)}.")
        logger.info(f"Total number of validation data is {len(valid_files)}.")
    
    data_transform = prepare_transform(include_body_region=include_body_region)
    
    workers_per_gpu = args.cpus_per_task // world_size
    train_dataset = CacheDataset(data=train_files, transform=data_transform, cache_rate=args.cache_rate, num_workers=workers_per_gpu)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=workers_per_gpu, 
        sampler=train_sampler, 
        collate_fn=custom_collate, ###
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    steps_per_epoch = len(train_loader)
    
    val_total = len(valid_files)
    per_rank = (val_total + world_size - 1) // world_size  # ceil
    start = rank * per_rank
    end = min(val_total, start + per_rank)
    val_files_shard = valid_files[start:end]
    dataset_val = CacheDataset(data=val_files_shard, transform=data_transform, cache_rate=0.0, num_workers=workers_per_gpu)
    valid_loader = DataLoader(
        dataset_val, 
        batch_size=args.val_batch_size, 
        num_workers=workers_per_gpu, 
        collate_fn=custom_collate, ###
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    scale_factor = calculate_scale_factor(train_loader, device)
    scale_factor_cpu = scale_factor.to("cpu")
    #optimizer = torch.optim.Adam(params=unet.parameters(), lr=args.lr, fused=True)
    optimizer = bnb_optim.AdamW8bit(params=unet.parameters(), lr=args.lr)
    total_opt_steps = (args.max_train_steps - 100000 + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps #### check pretraining totalsteps when submitting fine-tuning job
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_opt_steps, power=2.0)
    scaler = GradScaler() if args.amp else None
    if args.loss_type == "l1":
        loss_pt = torch.nn.L1Loss()
    elif args.loss_type == "l2":
        loss_pt = torch.nn.MSELoss()
    aux_loss_fn = FrequencyWeightedLoss(device="cpu")
    
    start_step, best_val_loss, start_epoch = 0, float("inf"), 0
    if args.resume:
        start_step, best_val_loss, start_epoch = resume_from_latest(
            unet, optimizer, lr_scheduler, scaler, args.model_dir, device)
    dist.barrier(device_ids=[local_rank])
    
    if rank == 0: logger.info("Compiling models with torch.compile()...")
    unet = torch.compile(unet)
    dist.barrier(device_ids=[local_rank])
    unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=False)

    sync_vec = torch.tensor([start_step, float(best_val_loss)], device=device, dtype=torch.float32)
    dist.broadcast(sync_vec, src=0)
    start_step, best_val_loss = int(sync_vec[0].item()), float(sync_vec[1].item())

    # count params
    if rank == 0:
        logger.info(f"### start_epoch: {start_epoch}")
        param_counts = count_parameters(unet)
        logger.info(f"### UNET's Trainable parameters: {param_counts['trainable']:,}")

    def infinite_loader(loader, sampler, start_epoch=0):
        epoch = start_epoch
        while True:
            sampler.set_epoch(epoch) # DDP CHANGE: 매 에포크마다 샘플러 시드 변경
            for batch in loader:
                yield batch
            epoch += 1

    train_iter = infinite_loader(train_loader, train_sampler, start_epoch)
    progress_bar = trange(start_step, args.max_train_steps + 1,
                          desc=f"Training on Rank {rank}",
                          initial=start_step, total=args.max_train_steps + 1,
                          disable=(rank != 0))

    for step in progress_bar:
        unet.train()
        batch = next(train_iter)
        latents = batch["latent"].to(device, non_blocking=True).contiguous() * scale_factor

        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        if include_modality:
            modality_tensor = torch.ones((len(latents),), dtype=torch.long).to(device)
        spacing_tensor = batch["spacing"].to(device, non_blocking=True)
        meta_tensor = batch["cond"].to(device, non_blocking=True) ### 👈

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            noise = torch.randn_like(latents)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(latents)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (latents.shape[0],), device=latents.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)

            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
                "meta_tensor": meta_tensor ### 👈
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

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                # predict noise
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                # predict sample
                model_gt = latents
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                # predict velocity
                model_gt = latents - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            latent_loss = loss_pt(model_output.float(), model_gt.float())

            # 보조 손실 (Voxel Space, 확률적) ###
            # 배치에서 무작위로 하나의 인덱스 선택
            total_loss = args.latent_weight * latent_loss
            spatial_loss, low_wavelet_loss, high_wavelet_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

            # idx = torch.randint(0, model_output.shape[0], (1,)).item()
            idx = batch["target_voxel_idx"]
            t_sample = timesteps[idx:idx+1]
            dist.broadcast(t_sample, src=0)
            if t_sample.item() < args.hwl_timestep_threshold:
                v_pred_sample = model_output[idx:idx+1]
                x_t_sample = noisy_latent[idx:idx+1]
                
                # predicted_x0_latent 계산 (Velocity 예측 모델용)
                t_normalized_sample = t_sample.float() / noise_scheduler.num_train_timesteps
                t_reshaped_norm = t_normalized_sample.view(-1, *([1] * (x_t_sample.dim() - 1)))
                predicted_x0_latent_gpu = x_t_sample + t_reshaped_norm * v_pred_sample
        
        if 'predicted_x0_latent_gpu' in locals() and t_sample.item() < args.hwl_timestep_threshold:
            predicted_x0_latent_cpu = predicted_x0_latent_gpu.to("cpu").float()
            target_x0_voxel_sample = batch["target_voxel_image"].float().unsqueeze(0)

            # 3. (핵심) VAE Decoder를 CPU에서 실행
            predicted_x0_voxel_cpu = autoencoder.decode_stage_2_outputs(
                predicted_x0_latent_cpu / scale_factor_cpu
            )
            # 4. (핵심) DWT 및 L1 Loss를 CPU에서 실행
            spatial_loss, low_wavelet_loss, high_wavelet_loss = aux_loss_fn(
                predicted_x0_voxel_cpu, target_x0_voxel_sample
            )
            # 5. 최종 손실 결합 (CPU 텐서 -> GPU 텐서로 이동)
            hwl_loss = (args.voxel_spatial_weight * spatial_loss + \
                        args.voxel_low_wavelet_weight * low_wavelet_loss + \
                        args.voxel_high_wavelet_weight * high_wavelet_loss)
            
            total_loss += hwl_loss.to(device) # 👈 최종 스칼라 값만 GPU로 다시 이동
            
            # 다음 루프를 위해 변수 삭제 (선택적이지만 권장)
            del predicted_x0_latent_gpu, predicted_x0_latent_cpu, target_x0_voxel_sample, predicted_x0_voxel_cpu
            
        loss = total_loss / args.gradient_accumulation_steps

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_total = reduce_mean_scalar(loss) * args.gradient_accumulation_steps
        latent_loss = reduce_mean_scalar(latent_loss) * args.gradient_accumulation_steps
        spatial_loss = reduce_mean_scalar(spatial_loss.to(device)) * args.gradient_accumulation_steps
        low_wavelet_loss = reduce_mean_scalar(low_wavelet_loss.to(device)) * args.gradient_accumulation_steps
        high_wavelet_loss = reduce_mean_scalar(high_wavelet_loss.to(device)) * args.gradient_accumulation_steps

        if rank == 0:
            progress_bar.set_postfix({'Total_loss': f"{loss_total:.4f}"})
            if args.report_to and step % 100 == 0: # 로그 기록 빈도 조절
                log_data = {
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/loss_total": loss_total,
                    "train/latent_loss": latent_loss,
                    "train/voxel_spatial_loss": spatial_loss,
                    "train/voxel_low_wavelet_loss": low_wavelet_loss,
                    "train/voxel_high_wavelet_loss": high_wavelet_loss
                }
                wandb.log(log_data, step=step)

        did_validate = False
        if (step % args.validation_steps == 0 or step == args.max_train_steps) and step > start_step:
            did_validate = True
            unet.eval()
            val_epoch_loss = {"latent_loss": 0, 
                              "voxel_spatial_loss": 0,
                              "voxel_low_wavelet_loss": 0,
                              "voxel_high_wavelet_loss": 0}
            num_val_batches_local = 0
            
            with torch.no_grad():
                for val_batch in valid_loader:
                    latents = val_batch["latent"].to(device, non_blocking=True) * scale_factor
                    spacing_tensor = val_batch["spacing"].to(device, non_blocking=True)
                    meta_tensor = val_batch["cond"].to(device, non_blocking=True) ### 👈
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                        noisy_latent = noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)

                        unet_inputs = {
                            "x": noisy_latent,
                            "timesteps": timesteps,
                            "spacing_tensor": spacing_tensor,
                            "meta_tensor": meta_tensor
                        }

                        if include_body_region:
                            top_region_index_tensor = val_batch["top_region_index"].to(device)
                            bottom_region_index_tensor = val_batch["bottom_region_index"].to(device)
                            unet_inputs.update({
                                "top_region_index_tensor": top_region_index_tensor,
                                "bottom_region_index_tensor": bottom_region_index_tensor,
                            })
                        if include_modality:
                            modality_tensor = torch.ones((len(latents),), dtype=torch.long).to(device)
                            unet_inputs.update({"class_labels": modality_tensor})

                        model_output = unet.module(**unet_inputs)

                        if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                            model_gt = noise
                        elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                            model_gt = latents
                        elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                            model_gt = latents - noise
                        else:
                            raise ValueError(
                                "noise scheduler prediction type has to be chosen from "
                                f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]"
                            )
                        
                        val_epoch_loss["latent_loss"] += loss_pt(model_output.float(), model_gt.float())

                        # 보조 손실 (Voxel Space, 확률적)
                        # 배치에서 무작위로 하나의 인덱스 선택
                        # idx = torch.randint(0, model_output.shape[0], (1,)).item()
                        idx = val_batch["target_voxel_idx"]
                        t_sample = timesteps[idx:idx+1]
                        dist.broadcast(t_sample, src=0)
                        if t_sample.item() < args.hwl_timestep_threshold:
                            v_pred_sample = model_output[idx:idx+1]
                            x_t_sample = noisy_latent[idx:idx+1]
                            
                            # predicted_x0_latent 계산 (Velocity 예측 모델용)
                            t_normalized_sample = t_sample.float() / noise_scheduler.num_train_timesteps
                            t_reshaped_norm = t_normalized_sample.view(-1, *([1] * (x_t_sample.dim() - 1)))
                            val_predicted_x0_latent_gpu = x_t_sample + t_reshaped_norm * v_pred_sample
                    
                    if 'val_predicted_x0_latent_gpu' in locals() and t_sample.item() < args.hwl_timestep_threshold:
                        predicted_x0_latent_cpu = val_predicted_x0_latent_gpu.to("cpu").float()
                        target_x0_voxel_sample = val_batch["target_voxel_image"].float().unsqueeze(0)

                        # 3. (핵심) VAE Decoder를 CPU에서 실행
                        predicted_x0_voxel_cpu = autoencoder.decode_stage_2_outputs(
                            predicted_x0_latent_cpu / scale_factor_cpu
                        )
                        # 4. (핵심) DWT 및 L1 Loss를 CPU에서 실행
                        spatial_loss, low_wavelet_loss, high_wavelet_loss = aux_loss_fn(
                            predicted_x0_voxel_cpu, target_x0_voxel_sample
                        )

                        val_epoch_loss["voxel_spatial_loss"] += spatial_loss.to(device)
                        val_epoch_loss["voxel_low_wavelet_loss"] += low_wavelet_loss.to(device)
                        val_epoch_loss["voxel_high_wavelet_loss"] += high_wavelet_loss.to(device)
                        del val_predicted_x0_latent_gpu, predicted_x0_latent_cpu, target_x0_voxel_sample, predicted_x0_voxel_cpu

                    num_val_batches_local += 1

            val_metrics = torch.tensor([val_epoch_loss["latent_loss"],
                                        val_epoch_loss["voxel_spatial_loss"],
                                        val_epoch_loss["voxel_low_wavelet_loss"],
                                        val_epoch_loss["voxel_high_wavelet_loss"],
                                        num_val_batches_local], device=device)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
            total_batches = val_metrics[-1].item()
            avg_latent_loss = val_metrics[0].item() / total_batches if total_batches > 0 else 0
            avg_voxel_spatial_loss = val_metrics[1].item() / total_batches if total_batches > 0 else 0
            avg_voxel_low_wavelet_loss = val_metrics[2].item() / total_batches if total_batches > 0 else 0
            avg_voxel_high_wavelet_loss = val_metrics[3].item() / total_batches if total_batches > 0 else 0
            final_val_losses = {"latent_loss": avg_latent_loss, 
                                "voxel_spatial_loss": avg_voxel_spatial_loss, 
                                "voxel_low_wavelet_loss": avg_voxel_low_wavelet_loss,
                                "voxel_high_wavelet_loss": avg_voxel_high_wavelet_loss}
            val_loss_g = args.latent_weight * final_val_losses["latent_loss"] + \
                args.voxel_spatial_weight * final_val_losses["voxel_spatial_loss"] + \
                args.voxel_low_wavelet_weight * final_val_losses["voxel_low_wavelet_loss"] + \
                args.voxel_high_wavelet_weight * final_val_losses["voxel_high_wavelet_loss"]
            #val_loss = {"loss": avg_loss}

            if rank == 0:
                logger.info(f"\nStep {step} Total Val Loss (Avg across all ranks): {val_loss_g:.4f}, Details: {final_val_losses}")
                if args.report_to:
                    log_data = {
                        "valid/total_loss": val_loss_g, 
                        "valid/scale_factor": scale_factor,
                        "valid/latent_loss": final_val_losses["latent_loss"],
                        "valid/voxel_spatial_loss": final_val_losses["voxel_spatial_loss"],
                        "valid/voxel_low_wavelet_loss": final_val_losses["voxel_low_wavelet_loss"],
                        "valid/voxel_high_wavelet_loss": final_val_losses["voxel_high_wavelet_loss"],
                    }
                    wandb.log(log_data, step=step)

                if val_loss_g < best_val_loss:
                    torch.cuda.synchronize(device)
                    unet_state_dict = unet.module._orig_mod.state_dict()
                    current_epoch = step // steps_per_epoch
                    best_val_loss = float(val_loss_g)
                    state = {
                        "unet_state_dict": unet_state_dict,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "num_train_timesteps": num_train_timesteps,
                        "scale_factor": scale_factor,
                        "epoch": current_epoch
                    }
                    if args.amp:
                        state["scaler"] = scaler.state_dict()
                    
                    best_dir = os.path.join(args.model_dir, "best-checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    atomic_save(state, os.path.join(best_dir, "diff_unet_ckpt.pt"))
                    logger.info(f"[best] updated at step {step}: {best_val_loss:.6f}")
                else:
                    logger.info(f"[not best] not updated at step {step}: {val_loss_g:.6f}")

            _best = torch.tensor([best_val_loss], device=device, dtype=torch.float32)
            dist.broadcast(_best, src=0)
            best_val_loss = float(_best.item())

        # --- 체크포인트 저장 ---
        is_time_to_save = (step % args.checkpointing_steps == 0 and step > start_step)
        if (is_time_to_save or SHUTDOWN_REQUESTED) and rank == 0:
            torch.cuda.synchronize(device)
            unet_state_dict = unet.module._orig_mod.state_dict()
            current_epoch = step // steps_per_epoch
            state = {
                "unet_state_dict": unet_state_dict,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "step": step,
                "best_val_loss": float(best_val_loss),
                "num_train_timesteps": num_train_timesteps,
                "scale_factor": scale_factor,
                "epoch": current_epoch
            }
            if args.amp:
                state["scaler"] = scaler.state_dict()
            
            ckpt_dir = os.path.join(args.model_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            atomic_save(state, os.path.join(ckpt_dir, "diff_unet_ckpt.pt"))
            logger.info(f"\nSaved Step {step} checkpoint to {ckpt_dir}")

        shutdown_tensor = torch.tensor([1 if (rank == 0 and SHUTDOWN_REQUESTED) else 0], device=device)
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor.item() == 1:
            if rank == 0:
                logger.info("Shutdown signal received and synced across all ranks. Exiting training loop gracefully.")
            break
        if did_validate:
            dist.barrier(device_ids=[local_rank])

    if SHUTDOWN_REQUESTED:
        print("Graceful shutdown initiated, exiting with code 1 to trigger requeue.")
        sys.exit(1) # ✅ 0 대신 1로 변경하여 "실패" 신호 전송
    
    dist.barrier(device_ids=[local_rank]) # DDP CHANGE: 모든 프로세스가 끝날 때까지 대기 후 정리
    cleanup_ddp()

if __name__ == '__main__':
    main()