#!/usr/bin/env python3
"""
测试混合精度修复
"""

import torch
from diffusers import VQModel
from accelerate import Accelerator

def test_mixed_precision_fix():
    print("🧪 测试混合精度修复...")
    
    # 初始化accelerator with fp16
    accelerator = Accelerator(mixed_precision="fp16")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # 加载VAE模型
    print("加载VAE模型...")
    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    
    # 准备模型（这会应用混合精度）
    vae = accelerator.prepare(vae)
    vae.requires_grad_(False)
    
    # 创建测试数据
    print("创建测试数据...")
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 256, 256)
    
    print(f"原始图像数据类型: {test_images.dtype}")
    print(f"VAE权重数据类型: {next(vae.parameters()).dtype}")
    
    # 测试修复后的编码逻辑
    print("测试VAE编码...")
    try:
        with torch.no_grad():
            pixel_values = test_images.to(accelerator.device)
            print(f"移动到设备后数据类型: {pixel_values.dtype}")
            
            # 应用混合精度修复
            if accelerator.mixed_precision == "fp16":
                pixel_values = pixel_values.half()
                print(f"转换为fp16后数据类型: {pixel_values.dtype}")
            
            latents = vae.encode(pixel_values).latents
            latents = latents * 0.18215
            
            print(f"✅ VAE编码成功！")
            print(f"潜在空间形状: {latents.shape}")
            print(f"潜在空间数据类型: {latents.dtype}")
            
    except Exception as e:
        print(f"❌ VAE编码失败: {e}")
        return False
    
    print("🎉 混合精度修复测试通过！")
    return True

if __name__ == "__main__":
    test_mixed_precision_fix()
