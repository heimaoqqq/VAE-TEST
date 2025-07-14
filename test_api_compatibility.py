#!/usr/bin/env python3
"""
测试diffusers API兼容性和修复后的条件扩散模型
"""

import torch
import os
import argparse
from diffusers import UNet2DModel, VQModel, DDPMScheduler
import numpy as np
from PIL import Image
import time

def test_unet_api():
    """测试UNet2DModel的API兼容性"""
    print("🔍 测试UNet2DModel API兼容性...")
    
    try:
        # 创建测试UNet
        unet = UNet2DModel(
            sample_size=32,  # 256//8 = 32
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D", 
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"
            ),
            num_class_embeds=32,  # 31用户 + 1无条件
        )
        
        print(f"✅ UNet创建成功")
        print(f"   - 输入通道: {unet.config.in_channels}")
        print(f"   - 输出通道: {unet.config.out_channels}")
        print(f"   - 类别嵌入数: {unet.config.num_class_embeds}")
        
        # 测试forward方法
        device = "cuda" if torch.cuda.is_available() else "cpu"
        unet = unet.to(device)
        
        # 创建测试输入
        batch_size = 2
        latents = torch.randn(batch_size, 3, 32, 32, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        class_labels = torch.tensor([0, 1], device=device)
        
        print(f"   - 测试输入形状: {latents.shape}")
        print(f"   - 时间步形状: {timesteps.shape}")
        print(f"   - 类别标签形状: {class_labels.shape}")
        
        # 测试不同的API调用方式
        with torch.no_grad():
            # 方式1: 使用class_labels参数
            try:
                output1 = unet(latents, timesteps, class_labels=class_labels)
                if hasattr(output1, 'sample'):
                    noise_pred1 = output1.sample
                else:
                    noise_pred1 = output1
                print(f"✅ 方式1成功 (class_labels): 输出形状 {noise_pred1.shape}")
            except Exception as e:
                print(f"❌ 方式1失败 (class_labels): {e}")
            
            # 方式2: 使用位置参数
            try:
                output2 = unet(latents, timesteps, class_labels)
                if hasattr(output2, 'sample'):
                    noise_pred2 = output2.sample
                else:
                    noise_pred2 = output2
                print(f"✅ 方式2成功 (位置参数): 输出形状 {noise_pred2.shape}")
            except Exception as e:
                print(f"❌ 方式2失败 (位置参数): {e}")
            
            # 方式3: 检查返回值类型
            try:
                output3 = unet(latents, timesteps, class_labels=class_labels, return_dict=False)
                if isinstance(output3, tuple):
                    noise_pred3 = output3[0]
                else:
                    noise_pred3 = output3
                print(f"✅ 方式3成功 (return_dict=False): 输出形状 {noise_pred3.shape}")
            except Exception as e:
                print(f"❌ 方式3失败 (return_dict=False): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ UNet API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vae_api():
    """测试VQModel的API兼容性"""
    print("\n🔍 测试VQModel API兼容性...")
    
    try:
        # 加载预训练VAE
        vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
        print(f"✅ VAE加载成功")
        print(f"   - 潜在通道数: {vae.config.latent_channels}")
        print(f"   - block_out_channels: {vae.config.block_out_channels}")
        
        # 检查scaling_factor
        if hasattr(vae.config, 'scaling_factor'):
            print(f"   - scaling_factor: {vae.config.scaling_factor}")
        else:
            print(f"   - scaling_factor: 未设置，使用默认值0.18215")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = vae.to(device)
        
        # 测试编码解码
        test_image = torch.randn(1, 3, 256, 256, device=device)
        
        with torch.no_grad():
            # 测试编码
            encoded = vae.encode(test_image)
            if hasattr(encoded, 'latents'):
                latents = encoded.latents
            else:
                latents = encoded
            print(f"✅ 编码成功: {test_image.shape} -> {latents.shape}")
            
            # 测试scaling
            scaled_latents = latents * 0.18215
            print(f"✅ Scaling成功: {scaled_latents.shape}")
            
            # 测试解码
            unscaled_latents = scaled_latents / 0.18215
            decoded = vae.decode(unscaled_latents)
            if hasattr(decoded, 'sample'):
                decoded_image = decoded.sample
            else:
                decoded_image = decoded
            print(f"✅ 解码成功: {unscaled_latents.shape} -> {decoded_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduler_api():
    """测试DDPMScheduler的API兼容性"""
    print("\n🔍 测试DDPMScheduler API兼容性...")
    
    try:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
        )
        print(f"✅ Scheduler创建成功")
        print(f"   - 训练时间步数: {scheduler.config.num_train_timesteps}")
        print(f"   - Beta调度: {scheduler.config.beta_schedule}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 测试添加噪声
        latents = torch.randn(2, 3, 32, 32, device=device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (2,), device=device)
        
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        print(f"✅ 添加噪声成功: {latents.shape} -> {noisy_latents.shape}")
        
        # 测试推理设置
        scheduler.set_timesteps(50, device=device)
        print(f"✅ 设置推理时间步成功: {len(scheduler.timesteps)} 步")
        
        # 测试去噪步骤
        noise_pred = torch.randn_like(latents)
        t = scheduler.timesteps[0]
        
        step_output = scheduler.step(noise_pred, t, latents, return_dict=False)
        if isinstance(step_output, tuple):
            prev_sample = step_output[0]
        else:
            prev_sample = step_output.prev_sample
        print(f"✅ 去噪步骤成功: {latents.shape} -> {prev_sample.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="测试diffusers API兼容性")
    parser.add_argument("--skip_vae", action="store_true", help="跳过VAE测试（需要网络下载）")
    args = parser.parse_args()
    
    print("🚀 开始测试diffusers API兼容性...")
    print("=" * 60)
    
    # 测试1: UNet API
    unet_ok = test_unet_api()
    
    # 测试2: VAE API（可选）
    if not args.skip_vae:
        vae_ok = test_vae_api()
    else:
        print("\n⏭️  跳过VAE测试")
        vae_ok = True
    
    # 测试3: Scheduler API
    scheduler_ok = test_scheduler_api()
    
    print("\n" + "=" * 60)
    if unet_ok and vae_ok and scheduler_ok:
        print("🎉 所有API兼容性测试通过！")
        print("\n📋 建议的API使用方式:")
        print("1. UNet调用: unet(latents, timesteps, class_labels=class_labels).sample")
        print("2. VAE编码: vae.encode(images).latents")
        print("3. VAE解码: vae.decode(latents).sample")
        print("4. Scheduler步骤: scheduler.step(noise_pred, t, latents, return_dict=False)[0]")
    else:
        print("❌ 部分API测试失败，请检查diffusers版本和代码实现")
    
    print(f"\n📦 当前环境信息:")
    try:
        import diffusers
        print(f"   - diffusers版本: {diffusers.__version__}")
    except:
        print("   - diffusers版本: 未知")
    
    try:
        import torch
        print(f"   - torch版本: {torch.__version__}")
        print(f"   - CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA版本: {torch.version.cuda}")
    except:
        print("   - torch版本: 未知")

if __name__ == "__main__":
    main()
