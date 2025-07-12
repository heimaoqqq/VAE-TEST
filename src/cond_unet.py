import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from typing import List, Optional, Tuple, Union

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
            user_embed = self.user_embedding(user_ids)  # [batch_size, user_embed_dim]
            
        # 使用原始UNet的时间嵌入部分
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
            
        # 确保时间步与批次大小匹配
        timesteps = timesteps.expand(sample.shape[0])
        
        t_emb = self.unet.time_proj(timesteps)
        t_emb = self.unet.time_embedding(t_emb)
        
        # 现在我们需要将条件注入到UNet的各个阶段
        # 这需要重新实现UNet的前向传播流程
        
        # 1. 初始卷积
        x = self.unet.conv_in(sample)
        
        # 2. 下采样阶段
        down_block_res_samples = (x,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
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
        x = self.unet.mid_block(x, t_emb)
        if user_embed is not None:
            x = self.mid_film_layer(x, user_embed)
            
        # 4. 上采样阶段
        for i, upsample_block in enumerate(self.unet.up_blocks):
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
        x = self.unet.conv_norm_out(x)
        x = self.unet.conv_act(x)
        x = self.unet.conv_out(x)
        
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