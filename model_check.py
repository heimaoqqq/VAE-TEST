import os
import torch
import json
import argparse
from pathlib import Path
from diffusers import VQModel, UNet2DModel
from src.cond_unet import CondUNet2DModel
from src.cond_pipeline import CondLatentDiffusionPipeline

def check_model_structure(model_path):
    """检查模型结构和权重是否正确"""
    print(f"\n===== 检查模型结构: {model_path} =====")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径 {model_path} 不存在!")
        return False
    
    # 检查组件目录
    required_components = ["vae", "unet", "scheduler"]
    for component in required_components:
        component_path = os.path.join(model_path, component)
        if not os.path.exists(component_path):
            print(f"错误: 组件目录 {component} 不存在!")
            return False
        print(f"✓ 找到组件目录: {component}")
    
    # 检查model_index.json
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(model_index_path):
        print(f"错误: model_index.json 不存在!")
        return False
    
    try:
        with open(model_index_path, "r") as f:
            model_index = json.load(f)
        print(f"✓ model_index.json 加载成功: {model_index}")
        
        # 检查_class_name
        if model_index.get("_class_name") != "CondLatentDiffusionPipeline":
            print(f"警告: model_index.json 中的 _class_name 不是 CondLatentDiffusionPipeline")
        
        # 检查组件列表
        components = model_index.get("components", [])
        for component in required_components:
            if component not in components:
                print(f"警告: model_index.json 中缺少组件 {component}")
    except Exception as e:
        print(f"错误: 无法解析 model_index.json: {e}")
        return False
    
    # 检查UNet结构
    unet_path = os.path.join(model_path, "unet")
    try:
        # 检查配置文件
        unet_config_path = os.path.join(unet_path, "config.json")
        if not os.path.exists(unet_config_path):
            print(f"错误: UNet配置文件不存在!")
            return False
        
        with open(unet_config_path, "r") as f:
            unet_config = json.load(f)
        print(f"✓ UNet配置加载成功")
        
        # 检查用户嵌入维度
        user_embed_dim = unet_config.get("user_embed_dim")
        if user_embed_dim is None:
            print(f"警告: UNet配置中缺少user_embed_dim")
        else:
            print(f"✓ 用户嵌入维度: {user_embed_dim}")
        
        # 检查用户数量
        num_users = unet_config.get("num_users")
        if num_users is None:
            print(f"警告: UNet配置中缺少num_users")
        else:
            print(f"✓ 用户数量: {num_users}")
        
        # 检查base_unet目录
        base_unet_path = os.path.join(unet_path, "base_unet")
        if not os.path.exists(base_unet_path):
            print(f"错误: base_unet目录不存在!")
            return False
        print(f"✓ 找到base_unet目录")
        
        # 检查pytorch_model.bin
        pytorch_model_path = os.path.join(unet_path, "pytorch_model.bin")
        if not os.path.exists(pytorch_model_path):
            print(f"错误: pytorch_model.bin不存在!")
            return False
        
        # 加载权重检查大小
        state_dict = torch.load(pytorch_model_path, map_location="cpu")
        print(f"✓ pytorch_model.bin加载成功，包含 {len(state_dict)} 个键")
        
        # 检查用户嵌入权重
        if "user_embedding.weight" not in state_dict:
            print(f"错误: 权重中缺少user_embedding.weight!")
            return False
        
        user_embed_shape = state_dict["user_embedding.weight"].shape
        print(f"✓ 用户嵌入形状: {user_embed_shape}")
        
        # 检查FiLM层
        film_keys = [k for k in state_dict.keys() if "film" in k.lower()]
        if not film_keys:
            print(f"错误: 权重中缺少FiLM层!")
            return False
        print(f"✓ 找到 {len(film_keys)} 个FiLM层相关权重")
        
    except Exception as e:
        print(f"错误: 检查UNet结构时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 尝试加载模型
    try:
        print("\n尝试加载模型组件...")
        
        # 加载VAE
        print("加载VAE...")
        vae = VQModel.from_pretrained(os.path.join(model_path, "vae"))
        print(f"✓ VAE加载成功")
        
        # 加载UNet
        print("加载条件UNet...")
        cond_unet = CondUNet2DModel.from_pretrained(os.path.join(model_path, "unet"))
        print(f"✓ 条件UNet加载成功")
        
        # 检查用户嵌入
        if not hasattr(cond_unet, "user_embedding"):
            print(f"错误: 加载的条件UNet缺少user_embedding!")
            return False
        
        print(f"✓ 用户嵌入形状: {cond_unet.user_embedding.weight.shape}")
        
        # 检查FiLM层
        if not hasattr(cond_unet, "down_film_layers") or len(cond_unet.down_film_layers) == 0:
            print(f"错误: 加载的条件UNet缺少down_film_layers!")
            return False
        
        print(f"✓ 下采样FiLM层数量: {len(cond_unet.down_film_layers)}")
        print(f"✓ 上采样FiLM层数量: {len(cond_unet.up_film_layers)}")
        
    except Exception as e:
        print(f"错误: 加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n===== 模型检查完成，结构正确! =====")
    return True

def main():
    parser = argparse.ArgumentParser(description='检查条件扩散模型结构')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    args = parser.parse_args()
    
    check_model_structure(args.model_path)

if __name__ == "__main__":
    main() 
