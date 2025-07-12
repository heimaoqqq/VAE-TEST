import argparse
import logging
import math
import os
import random
import shutil
import warnings
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

from src.cond_unet import CondUNet2DModel
from src.cond_pipeline import CondLatentDiffusionPipeline

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="训练条件潜在扩散模型")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="预训练模型的路径，训练好的无条件模型",
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
        "--eval_batch_size", 
        type=int, 
        default=8, 
        help="评估批次大小"
    )
    parser.add_argument(
        "--num_train_epochs", 
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
        default=0.1,
        help="无条件训练概率，模型学习在没有用户ID时也能生成图像",
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
        "--save_images_epochs", 
        type=int, 
        default=10, 
        help="每多少轮保存样本图像"
    )
    parser.add_argument(
        "--save_model_epochs", 
        type=int, 
        default=10, 
        help="每多少轮保存模型"
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=31,
        help="总用户数量",
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=50,
        help="评估时的推理步数",
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
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        
        # 设置转换
        self._transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # 收集所有图像文件并按用户ID分组
        self.image_paths = []
        self.user_ids = []
        
        for user_id in range(31):  # 假设有31位用户
            user_dir = self.data_dir / f"user_{user_id}"
            if not user_dir.exists():
                print(f"警告: 用户{user_id}目录不存在 {user_dir}")
                continue
                
            # 获取该用户的所有图像文件
            user_images = list(user_dir.glob("*.png"))
            if not user_images:
                user_images = list(user_dir.glob("*.jpg"))
            
            if not user_images:
                print(f"警告: 用户{user_id}没有图像文件")
                continue
                
            self.image_paths.extend(user_images)
            self.user_ids.extend([user_id] * len(user_images))
            
        if not self.image_paths:
            raise RuntimeError(f"在{data_dir}中找不到任何图像")
            
        print(f"加载了{len(self.image_paths)}张图像，来自{len(set(self.user_ids))}个用户")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        user_id = self.user_ids[idx]
        
        image = transforms.ToPILImage()(np.array(transforms.ToTensor()(transforms.Image.open(image_path))))
        
        if self._transform is not None:
            image = self._transform(image)
            
        return {
            "input": image,
            "user_id": user_id,
        }


def main():
    args = parse_args()
    
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # 创建日志器
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        
    # 处理输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # 加载预训练模型
    logger.info(f"加载预训练模型组件: {args.pretrained_model_path}")
    
    # 先加载无条件模型获取基本组件
    from src.pipeline import UncondLatentDiffusionPipeline
    temp_pipeline = UncondLatentDiffusionPipeline.from_pretrained(args.pretrained_model_path)
    
    # 提取组件
    vae = temp_pipeline.vae
    scheduler = temp_pipeline.scheduler
    base_unet = temp_pipeline.unet
    
    # 释放临时pipeline
    del temp_pipeline
    
    # 创建条件UNet
    logger.info(f"创建条件UNet模型，用户嵌入维度: {args.user_embed_dim}")
    unet = CondUNet2DModel(
        base_unet=base_unet, 
        num_users=args.num_users, 
        user_embed_dim=args.user_embed_dim,
        freeze_unet=False  # 不冻结基础UNet，让整个网络一起训练
    )
    
    # 设置调度器
    noise_scheduler = scheduler
    
    # 加载数据集
    train_dataset = MicroDopplerDataset(
        data_dir=args.dataset_path,
        resolution=args.resolution,
        center_crop=True,
    )
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # 设置优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "要使用 8-bit Adam，请先安装bitsandbytes: `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    params_to_optimize = unet.parameters()
    
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 设置学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # 准备加速器
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 设置EMA
    if args.use_ema:
        ema_unet = CondUNet2DModel(
            base_unet=base_unet,
            num_users=args.num_users,
            user_embed_dim=args.user_embed_dim
        )
        ema_unet.to(accelerator.device)
        ema_model = EMAModel(
            ema_unet.parameters(),
            decay=0.9999,
            model_cls=CondUNet2DModel, 
            model_config=ema_unet.config
        )
    
    # 设置xFormers优化
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xFormers未安装，无法启用内存高效注意力")
    
    # 启用TF32精度
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        
    # 设置训练总步数
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    
    # 设置进度条
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("训练步骤")
    
    global_step = 0
    
    # 从检查点恢复训练
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # 获取最新检查点
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            
        if path is None:
            accelerator.print(f"检查点'{args.resume_from_checkpoint}'未找到，从头开始训练")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"从检查点'{path}'恢复训练")
            path = os.path.join(args.output_dir, path)
            accelerator.load_state(path)
            global_step = int(path.split("-")[1])
            
            # 调整进度条
            resume_global_step = global_step * args.gradient_accumulation_steps
            progress_bar.update(resume_global_step)
    
    # 权重数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # 将VAE移动到设备上并设置为评估模式
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    
    # 训练循环
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 转换为潜在表示
                clean_images = batch["input"].to(weight_dtype)
                latents = vae.encode(clean_images).latents
                latents = latents.to(dtype=weight_dtype)
                latents = latents * 0.18215
                
                # 获取用户ID
                user_ids = batch["user_id"].to(accelerator.device)
                
                # 随机丢弃部分条件（对无条件生成能力建模）
                batch_size = clean_images.shape[0]
                uncond_mask = torch.rand(batch_size, device=accelerator.device) < args.uncond_prob
                if uncond_mask.any():
                    # 对于标记为无条件的样本，将user_ids设为None
                    user_ids_with_uncond = user_ids.clone()
                    user_ids_with_uncond[uncond_mask] = -1  # 使用-1表示无条件
                    user_ids = user_ids_with_uncond
                
                # 采样噪声
                noise = torch.randn_like(latents)
                
                # 采样时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), 
                    device=latents.device
                ).long()
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测噪声
                model_pred = unet(noisy_latents, timesteps, user_ids=user_ids).sample
                
                # 计算损失
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # 更新EMA模型
            if args.use_ema and accelerator.sync_gradients:
                ema_model.step(unet.parameters())
                
            # 更新进度条
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 保存检查点
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"在{save_path}保存检查点")
                        
                        # 删除旧检查点
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                for old_ckpt in checkpoints[:num_to_remove]:
                                    old_ckpt_path = os.path.join(args.output_dir, old_ckpt)
                                    shutil.rmtree(old_ckpt_path)
                                    logger.info(f"删除旧检查点{old_ckpt_path}")
                
                # 生成示例图像
                if accelerator.is_main_process:
                    if global_step % (args.save_images_epochs * len(train_dataloader)) == 0 or global_step == args.max_train_steps:
                        # 使用EMA模型或当前模型
                        if args.use_ema:
                            ema_model.store(unet.parameters())
                            ema_model.copy_to(unet.parameters())
                        
                        # 创建评估pipeline
                        pipeline = CondLatentDiffusionPipeline(
                            vae=vae,
                            unet=unet,
                            scheduler=scheduler,
                        )
                        
                        # 为每个用户ID生成样本（从0到4）
                        for eval_user_id in range(5):
                            # 创建用户ID张量
                            eval_user_ids = torch.tensor([eval_user_id] * args.eval_batch_size, device=accelerator.device)
                            
                            # 设置随机种子
                            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                            
                            # 生成样本
                            images = pipeline(
                                batch_size=args.eval_batch_size,
                                user_ids=eval_user_ids,
                                num_inference_steps=args.ddpm_num_inference_steps,
                                generator=generator,
                                output_type="numpy",
                                guidance_scale=3.0,  # 使用条件引导
                            ).images
                            
                            # 保存图像
                            images_processed = (images * 255).round().astype("uint8")
                            for i, image in enumerate(images_processed):
                                image_pil = transforms.ToPILImage()(image.transpose(2, 0, 1)/255.0)
                                image_pil.save(
                                    os.path.join(
                                        args.output_dir, 
                                        f"epoch_{epoch}_step_{global_step}_user_{eval_user_id}_sample_{i}.png"
                                    )
                                )
                        
                        # 恢复原始权重
                        if args.use_ema:
                            ema_model.restore(unet.parameters())
                
                # 保存模型
                if accelerator.is_main_process:
                    if global_step % (args.save_model_epochs * len(train_dataloader)) == 0 or global_step == args.max_train_steps:
                        # 使用EMA模型或当前模型
                        if args.use_ema:
                            ema_model.store(unet.parameters())
                            ema_model.copy_to(unet.parameters())
                            
                        # 获取unwrapped模型
                        unet_unwrapped = accelerator.unwrap_model(unet)
                        
                        # 保存条件UNet模型
                        base_unet_unwrapped = unet_unwrapped.unet
                        
                        # 保存模型组件
                        pipeline = CondLatentDiffusionPipeline(
                            vae=vae,
                            unet=unet_unwrapped,
                            scheduler=scheduler,
                        )
                        
                        # 保存pipeline
                        pipeline.save_pretrained(args.output_dir)
                        
                        # 恢复原始权重
                        if args.use_ema:
                            ema_model.restore(unet.parameters())
            
            # 检查是否完成训练
            if global_step >= args.max_train_steps:
                break
        
    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 使用EMA模型或当前模型
        if args.use_ema:
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
            
        # 获取unwrapped模型
        unet = accelerator.unwrap_model(unet)
        
        # 保存条件UNet模型
        pipeline = CondLatentDiffusionPipeline(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        
        # 保存pipeline
        pipeline.save_pretrained(args.output_dir)
        
        # 恢复原始权重
        if args.use_ema:
            ema_model.restore(unet.parameters())
    
    accelerator.end_training()

if __name__ == "__main__":
    main() 