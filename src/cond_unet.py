import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from typing import List, Optional, Tuple, Union
import os
import json

class FiLMLayer(nn.Module):
    """
    特征调制层 (Feature-wise Linear Modulation)
    用于基于条件生成微调特征图
    """
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.film_proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels * 2)
        )
        
    def forward(self, feature_map, condition_embed):
        """
        Args:
            feature_map: 形状为 [batch_size, channels, height, width] 的特征图
            condition_embed: 形状为 [batch_size, embed_dim] 的条件嵌入
        """
        # 生成gamma(scale)和beta(shift)参数
        film_params = self.film_proj(condition_embed)  # [batch_size, out_channels*2]
        scale, shift = torch.chunk(film_params, 2, dim=1)  # 各自 [batch_size, out_channels]
        
        # 调整维度以便于广播
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [batch_size, out_channels, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [batch_size, out_channels, 1, 1]
        
        # 应用FiLM调制: gamma * x + beta
        modulated = scale * feature_map + shift
        
        return modulated


class CondUNet2DModel(nn.Module):
    """
    条件UNet2D模型，通过FiLM层添加用户ID条件
    """
    def __init__(
        self, 
        base_unet: UNet2DModel, 
        num_users: int = 31, 
        user_embed_dim: int = 64,
        freeze_unet: bool = False
    ):
        super().__init__()
        self.unet = base_unet
        
        # 如果需要，冻结基础UNet参数
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
        
        # 用户ID嵌入层
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # 为每个下采样块创建FiLM层
        self.down_film_layers = nn.ModuleList()
        for ch in self.unet.config.block_out_channels:
            self.down_film_layers.append(FiLMLayer(user_embed_dim, ch))
            
        # 为中间块创建FiLM层
        self.mid_film_layer = FiLMLayer(
            user_embed_dim, 
            self.unet.config.block_out_channels[-1]
        )
        
        # 为每个上采样块创建FiLM层
        self.up_film_layers = nn.ModuleList()
        # 上采样块的通道数是反向的
        for ch in reversed(self.unet.config.block_out_channels):
            self.up_film_layers.append(FiLMLayer(user_embed_dim, ch))
            
        # 保存配置
        self.config = {
            "num_users": num_users,
            "user_embed_dim": user_embed_dim,
            "freeze_unet": freeze_unet
        }
        
    def save_pretrained(self, save_directory):
        """
        保存模型到指定目录
        
        Args:
            save_directory: 保存目录
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)
        
        # 保存基础UNet
        self.unet.save_pretrained(os.path.join(save_directory, "base_unet"))
        
        # 保存用户嵌入和FiLM层
        state_dict = {
            "user_embedding.weight": self.user_embedding.weight,
        }
        
        # 保存下采样FiLM层
        for i, layer in enumerate(self.down_film_layers):
            for name, param in layer.named_parameters():
                state_dict[f"down_film_layers.{i}.{name}"] = param
        
        # 保存中间FiLM层
        for name, param in self.mid_film_layer.named_parameters():
            state_dict[f"mid_film_layer.{name}"] = param
        
        # 保存上采样FiLM层
        for i, layer in enumerate(self.up_film_layers):
            for name, param in layer.named_parameters():
                state_dict[f"up_film_layers.{i}.{name}"] = param
        
        # 保存权重
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """
        从预训练模型加载
        
        Args:
            pretrained_model_path: 预训练模型路径
        """
        # 加载配置
        with open(os.path.join(pretrained_model_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # 加载基础UNet
        base_unet = UNet2DModel.from_pretrained(os.path.join(pretrained_model_path, "base_unet"))
        
        # 创建模型
        model = cls(
            base_unet=base_unet,
            num_users=config["num_users"],
            user_embed_dim=config["user_embed_dim"],
            freeze_unet=config["freeze_unet"]
        )
        
        # 加载权重
        state_dict = torch.load(
            os.path.join(pretrained_model_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=False)
        
        return model
            
    def forward(
        self, 
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        user_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        前向传播函数
        
        Args:
            sample: 输入张量 [batch_size, channels, height, width]
            timestep: 时间步
            user_ids: 用户ID张量 [batch_size]
            return_dict: 是否返回字典格式的结果
        """
        # 首先获取用户嵌入向量
        user_embed = None
        if user_ids is not None:
            # 过滤出有效的用户ID (>=0)，-1表示无条件生成
            valid_mask = user_ids >= 0
            if valid_mask.any():
                # 只为有效ID创建嵌入
                valid_ids = user_ids[valid_mask]
                valid_embeds = self.user_embedding(valid_ids)  # [valid_count, user_embed_dim]
                
                # 创建全零嵌入向量
                user_embed = torch.zeros(
                    (user_ids.shape[0], self.user_embedding.embedding_dim), 
                    device=user_ids.device, 
                    dtype=valid_embeds.dtype
                )
                
                # 将有效嵌入放回对应位置
                # 修改索引赋值操作，避免torch.compile的问题
                # 原始代码: user_embed[valid_mask] = valid_embeds
                # 使用scatter操作替代直接索引赋值
                indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                for i, idx in enumerate(indices):
                    user_embed[idx] = valid_embeds[i]
            
        # 使用原始UNet的时间嵌入部分
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
            
        # 确保时间步与批次大小匹配
        timesteps = timesteps.expand(sample.shape[0])
        
        # 安全访问time_proj和time_embedding，处理DataParallel的情况
        try:
            t_emb = self.unet.time_proj(timesteps)
            t_emb = self.unet.time_embedding(t_emb)
        except AttributeError:
            # 如果是DataParallel模型，尝试通过module访问
            if hasattr(self.unet, 'module'):
                t_emb = self.unet.module.time_proj(timesteps)
                t_emb = self.unet.module.time_embedding(t_emb)
            else:
                # 检查是否自己是DataParallel
                if isinstance(self.unet, nn.DataParallel) or isinstance(self.unet, nn.parallel.DistributedDataParallel):
                    t_emb = self.unet.module.time_proj(timesteps)
                    t_emb = self.unet.module.time_embedding(t_emb)
                else:
                    raise AttributeError("无法访问time_proj属性，模型结构可能有问题")
        
        # 现在我们需要将条件注入到UNet的各个阶段
        # 这需要重新实现UNet的前向传播流程
        
        # 安全访问UNet的组件
        def safe_access(obj, attr_name):
            try:
                return getattr(obj, attr_name)
            except AttributeError:
                if hasattr(obj, 'module'):
                    return getattr(obj.module, attr_name)
                raise
        
        # 1. 初始卷积
        conv_in = safe_access(self.unet, 'conv_in')
        x = conv_in(sample)
        
        # 2. 下采样阶段
        down_blocks = safe_access(self.unet, 'down_blocks')
        down_block_res_samples = (x,)
        for i, downsample_block in enumerate(down_blocks):
            if user_embed is not None and i < len(self.down_film_layers):
                # 正常处理下采样块
                x, res_samples = downsample_block(
                    hidden_states=x, 
                    temb=t_emb,
                )
                # 应用FiLM调制
                x = self.down_film_layers[i](x, user_embed)
            else:
                x, res_samples = downsample_block(hidden_states=x, temb=t_emb)
                
            down_block_res_samples += res_samples
            
        # 3. 中间层处理
        mid_block = safe_access(self.unet, 'mid_block')
        x = mid_block(x, t_emb)
        if user_embed is not None:
            x = self.mid_film_layer(x, user_embed)
            
        # 4. 上采样阶段
        up_blocks = safe_access(self.unet, 'up_blocks')
        for i, upsample_block in enumerate(up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            # 正常处理上采样块
            x = upsample_block(
                hidden_states=x,
                temb=t_emb,
                res_hidden_states_tuple=res_samples,
            )
            
            # 应用FiLM调制
            if user_embed is not None and i < len(self.up_film_layers):
                x = self.up_film_layers[i](x, user_embed)
                
        # 5. 最终输出
        conv_norm_out = safe_access(self.unet, 'conv_norm_out')
        conv_act = safe_access(self.unet, 'conv_act')
        conv_out = safe_access(self.unet, 'conv_out')
        
        x = conv_norm_out(x)
        x = conv_act(x)
        x = conv_out(x)
        
        if not return_dict:
            return (x,)
            
        # 返回与原始UNet兼容的输出格式
        return UNet2DModelOutput(sample=x)
        
        
class UNet2DModelOutput:
    """
    与原始UNet2DModel输出兼容的输出类
    """
    def __init__(self, sample):
        self.sample = sample 
