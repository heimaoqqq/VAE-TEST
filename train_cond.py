import argparse
import logging
import math
import os
import random
import shutil
import warnings
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    VQModel,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from src.cond_pipeline import CondLatentDiffusionPipeline

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="训练条件潜在扩散模型")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=False,  # 修改为可选参数
        help="预训练模型的路径，如不提供则从头训练",
    )
    parser.add_argument(
        "--pretrained_vqvae_model_name_or_path",
        type=str,
        default="CompVis/ldm-celebahq-256",
        help="预训练VQ-VAE模型的路径或Hugging Face Hub名称",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="数据集路径，包含多个用户的微多普勒时频图",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="模型保存路径",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="训练分辨率",
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16, 
        help="训练批次大小"
    )
    parser.add_argument(
        "--save_model_epochs", 
        type=int, 
        default=10, 
        help="每多少轮保存模型"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=100, 
        help="训练轮数"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="最大训练步数，如果设置，将覆盖num_train_epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="初始学习率",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="学习率调度器类型",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="学习率预热步数"
    )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="是否使用8-bit Adam优化器"
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="是否使用EMA模型平均"
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="Adam优化器beta1"
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="Adam优化器beta2"
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Adam优化器权重衰减"
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Adam优化器epsilon"
    )
    parser.add_argument(
        "--user_embed_dim",
        type=int,
        default=64,
        help="用户嵌入维度",
    )
    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.15,
        help="无条件训练概率，模型学习在没有用户ID时也能生成图像（推荐值：0.15-0.2）",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="是否使用混合精度训练",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="是否允许TF32精度",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="要报告的跟踪器",
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="用于分布式训练的本地进程排名"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help="每多少步保存一次检查点",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help="保存的检查点总数限制",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="如果指定，将从该检查点恢复训练",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="是否使用xFormers优化注意力计算"
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=10,
        help="每多少轮运行一次验证并生成图像",
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=31,
        help="总用户数量",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class MicroDopplerDataset(torch.utils.data.Dataset):
    """
    微多普勒时频图像数据集，用于条件生成
    """
    def __init__(
        self,
        data_dir,
        resolution=256,
        center_crop=False,
        is_main_process=True,  # 添加参数以确定是否为主进程
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.is_main_process = is_main_process # 保存状态
        
        # 设置转换
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # 收集所有图像文件并按用户ID分组
        self.image_paths = []
        self.user_ids = []
        
        # 适应ID_1到ID_31的文件夹结构
        for folder_id in range(1, 32):  # 从1到31
            user_dir = self.data_dir / f"ID_{folder_id}"
            if not user_dir.exists():
                if self.is_main_process:  # 只在主进程中打印警告
                    print(f"警告: 文件夹ID_{folder_id}不存在 {user_dir}")
                continue
                
            # 获取该用户的所有图像文件
            user_images = list(user_dir.glob("*.png"))
            if not user_images:
                user_images = list(user_dir.glob("*.jpg"))
            
            if not user_images:
                if self.is_main_process:  # 只在主进程中打印警告
                    print(f"警告: 文件夹ID_{folder_id}没有图像文件")
                continue
                
            self.image_paths.extend(user_images)
            
            # 用户ID从0开始，文件夹从1开始，所以要减1进行映射
            model_user_id = folder_id - 1
            self.user_ids.extend([model_user_id] * len(user_images))
            
        if not self.image_paths:
            raise RuntimeError(f"在{data_dir}中找不到任何图像")
        
        # 只在主进程中打印数据集信息    
        if self.is_main_process:
            print(f"加载了{len(self.image_paths)}张图像，来自{len(set(self.user_ids))}个用户")
            for folder_id in range(1, 32):
                model_user_id = folder_id - 1
                count = self.user_ids.count(model_user_id)
                if count > 0:
                    print(f"  - ID_{folder_id} (内部ID: {model_user_id}) 有 {count} 张图像")
        
        self.num_users = len(set(self.user_ids))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        user_id = self.user_ids[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        # 应用图像变换
        image = self.transform(image)

        return {"pixel_values": image, "user_ids": user_id}


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # 日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # 处理混合精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 加载模型: VAE, UNet, Scheduler
    # 1. VAE
    # 使用指定的VQ-VAE
    vqvae_path = os.path.join(args.pretrained_vqvae_model_name_or_path, "vqvae")
    if os.path.exists(vqvae_path):
         vae = VQModel.from_pretrained(vqvae_path)
    else:
         vae = VQModel.from_pretrained(args.pretrained_vqvae_model_name_or_path, subfolder="vqvae")
    
    # 2. Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )

    # 3. UNet (用支持条件的标准UNet替换CondUNet)
    # 使用与无条件训练脚本类似的UNet配置，但添加class-conditioning
    # 确保UNet的sample_size与VQVAE输出的潜在空间尺寸匹配
    latent_size = args.resolution // (2 ** (len(vae.config.block_out_channels) - 1))
    
    unet = UNet2DModel(
        sample_size=latent_size,
        in_channels=3,  # 修复：与无条件训练保持一致，使用固定的3通道
        out_channels=3,  # 修复：与无条件训练保持一致，使用固定的3通道
        num_class_embeds=args.num_users + 1,  # +1 用于无条件生成
    )

    # 冻结VAE
    vae.requires_grad_(False)
    
    # 启用xformers以节省内存
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            
            xformers_version = version.parse(xformers.__version__)
            if xformers_version >= version.parse("0.0.16"):
                logger.info("使用xFormers内存高效注意力机制")
                unet.enable_xformers_memory_efficient_attention()
            else:
                warnings.warn("xFormers版本过低，建议升级到0.0.16或更高版本")
        else:
            warnings.warn("未安装xFormers，无法启用内存高效注意力机制")
            
    # 优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("请先安装bitsandbytes: `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 数据集
    dataset = MicroDopplerDataset(
        data_dir=args.dataset_path,
        resolution=args.resolution,
        is_main_process=accelerator.is_main_process,
    )
    
    # 更新args.num_users以匹配数据集中找到的实际用户数
    if args.num_users != dataset.num_users:
        logger.info(f"更新用户数量从 {args.num_users} 到 {dataset.num_users}")
        args.num_users = dataset.num_users

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    # 学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs),
    )

    # EMA模型
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), 
            decay=0.9999, 
            model_cls=UNet2DModel, 
            model_config=unet.config
        )

    # 准备所有组件 - 包括VAE以确保混合精度处理正确
    unet, vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, optimizer, dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # 将模型移动到设备
    vae.to(accelerator.device, dtype=weight_dtype)

    # 计算总训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化跟踪器
    if accelerator.is_main_process:
        accelerator.init_trackers("train_cond_microdoppler", config=vars(args))

    # 开始训练
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** 开始训练 *****")
    logger.info(f"  总样本数 = {len(dataset)}")
    logger.info(f"  总轮数 = {args.num_epochs}")
    logger.info(f"  每个设备批次大小 = {args.train_batch_size}")
    logger.info(f"  总训练批次大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # 进度条
    

    for epoch in range(first_epoch, args.num_epochs):
        unet.train()
        train_loss = 0.0
        
        # 为每轮设置独立的进度条和计时
        epoch_progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        epoch_start_time = time.time()
        
        for step, batch in enumerate(epoch_progress_bar):
            with accelerator.accumulate(unet):
                # 将图像编码到潜在空间 - 确保数据类型匹配
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(accelerator.device)
                    # 如果使用混合精度，确保输入数据类型与VAE权重匹配
                    if accelerator.mixed_precision == "fp16":
                        pixel_values = pixel_values.half()
                    latents = vae.encode(pixel_values).latents

                # 修复：添加与无条件训练一致的scaling factor
                latents = latents * 0.18215

                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 获取用户ID
                user_ids = batch["user_ids"].to(accelerator.device)
                
                # 以一定概率进行无条件训练
                uncond_mask = torch.rand(user_ids.shape[0], device=user_ids.device) < args.uncond_prob
                # 无条件ID为num_users
                user_ids[uncond_mask] = args.num_users
                
                # 预测噪声
                noise_pred = unet(noisy_latents, timesteps, class_labels=user_ids).sample
                
                # 计算损失
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # 累积损失
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # 使用更温和的梯度裁剪
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 更新进度
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                global_step += 1

            # 更新进度条的描述
            epoch_progress_bar.set_postfix(loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"])

            if global_step >= args.max_train_steps:
                break
        
        # --- 每轮结束后的日志输出 ---
        if accelerator.is_main_process:
            epoch_duration = time.time() - epoch_start_time
            avg_epoch_loss = train_loss / len(dataloader)

            # 格式化时间输出
            hours, rem = divmod(epoch_duration, 3600)
            minutes, seconds = divmod(rem, 60)

            logger.info(
                f"Epoch {epoch + 1}/{args.num_epochs} | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Global Step: {global_step}"
            )

            # 记录到跟踪器
            accelerator.log({
                "train_loss": avg_epoch_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=global_step)
            
        accelerator.wait_for_everyone()

        # # Potentially validate during training
        # if accelerator.is_main_process:
        #     if args.validation_epochs is not None and epoch % args.validation_epochs == 0:
        #         logger.info("Running validation... ")
        # ... existing code ...
        #         logger.info(f"Saved validation images to {val_output_dir}")
                
        # accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()

        # Save the model
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())

            if (epoch + 1) % args.save_model_epochs == 0 or (epoch + 1) == args.num_epochs:
                unet_to_save = accelerator.unwrap_model(unet)
                
                # 创建并保存推理pipeline
                pipeline = CondLatentDiffusionPipeline(
                    unet=unet_to_save,
                    vqvae=vae,
                    scheduler=noise_scheduler,
                )
                
                save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
                pipeline.save_pretrained(save_dir)
                logger.info(f"Saved model checkpoint to {save_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
