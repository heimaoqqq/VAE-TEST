#!/usr/bin/env python3
"""
测试训练脚本修复 - 简化版本
"""

import torch
from diffusers import VQModel, UNet2DModel, DDPMScheduler
from accelerate import Accelerator
import os

def test_training_fix():
    print("🧪 测试训练脚本修复...")
    
    # 模拟训练参数
    class Args:
        mixed_precision = "fp16"
        
    args = Args()
    
    # 初始化accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    print(f"Mixed precision: {accelerator.mixed_precision}")
    
    try:
        # 加载模型
        print("加载模型...")
        vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
        
        # 创建简单的UNet用于测试
        unet = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        
        # 创建优化器和调度器
        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # 准备模型（应用混合精度）
        print("准备模型...")
        unet, vae, optimizer = accelerator.prepare(unet, vae, optimizer)
        
        # 冻结VAE
        vae.requires_grad_(False)
        
        print(f"VAE权重数据类型: {next(vae.parameters()).dtype}")
        print(f"UNet权重数据类型: {next(unet.parameters()).dtype}")
        
        # 创建测试数据
        print("创建测试数据...")
        batch_size = 2
        test_batch = {
            "pixel_values": torch.randn(batch_size, 3, 256, 256),
            "user_ids": torch.tensor([0, 1])
        }
        
        print("测试VAE编码...")
        # 测试修复后的VAE编码逻辑
        with torch.no_grad():
            pixel_values = test_batch["pixel_values"].to(accelerator.device)
            print(f"原始数据类型: {pixel_values.dtype}")
            
            # 应用混合精度修复
            if accelerator.mixed_precision == "fp16":
                pixel_values = pixel_values.half()
                print(f"转换后数据类型: {pixel_values.dtype}")
            
            latents = vae.encode(pixel_values).latents
            latents = latents * 0.18215
            
            print(f"✅ VAE编码成功！")
            print(f"潜在空间形状: {latents.shape}")
            print(f"潜在空间数据类型: {latents.dtype}")
        
        # 测试UNet前向传播
        print("测试UNet前向传播...")
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (batch_size,), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 创建条件
        user_ids = test_batch["user_ids"].to(accelerator.device)
        
        # UNet前向传播
        noise_pred = unet(noisy_latents, timesteps, class_labels=user_ids).sample
        
        print(f"✅ UNet前向传播成功！")
        print(f"噪声预测形状: {noise_pred.shape}")
        print(f"噪声预测数据类型: {noise_pred.dtype}")
        
        print("🎉 训练脚本修复测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_training_fix()
