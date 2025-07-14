import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Union, Tuple
from tqdm.auto import tqdm

from diffusers import (
    VQModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor

from src.pipeline import LatentDiffusionPipelineBase
from diffusers import UNet2DModel
import numpy as np


class CondLatentDiffusionPipeline(LatentDiffusionPipelineBase):
    """
    条件潜在扩散Pipeline，支持用户ID控制
    """
    def __init__(
            self,
            vqvae: VQModel,
            scheduler: Union[
                DDPMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler,
            ],
            unet: UNet2DModel,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        
        # 设置内部设备属性，基类的device属性会使用这个
        if hasattr(unet, 'device'):
            self._device = unet.device
        elif hasattr(vqvae, 'device'):
            self._device = vqvae.device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to(self, device=None, dtype=None):
        """
        将模型迁移到指定设备和数据类型
        
        Args:
            device: 目标设备，可以是字符串或torch.device对象
            dtype: 目标数据类型
        """
        # 首先调用基类的to方法，处理基本组件
        super().to(device, dtype)
        
        # 确保所有组件都被迁移到正确的设备
        if device is None:
            return self
        
        # 创建正确的torch.device对象
        if isinstance(device, str):
            device = torch.device(device)
        
        # 移动VAE
        if self.vqvae is not None:
            self.vqvae.to(device, dtype if dtype is not None else None)
            
        # 移动UNet (额外确认)
        if self.unet is not None:
            self.unet.to(device, dtype if dtype is not None else None)
        
        # 更新内部设备属性
        self._device = device
            
        return self
            
    def decode_latents(self, latents):
        """
        将潜在表示解码为图像
        
        Args:
            latents: 潜在表示 [B, C, H, W]
            
        Returns:
            numpy数组，形状为 [B, H, W, C]，值范围为[0, 1]
        """
        print(f"解码latents，形状: {latents.shape}, 类型: {latents.dtype}")

        # 修复：使用与训练一致的scaling factor
        latents = latents / 0.18215
        
        # 确保latents与vae的数据类型一致
        dtype = None
        if hasattr(self.vqvae, 'dtype'):
            dtype = self.vqvae.dtype
        elif hasattr(self.vqvae, 'encoder') and hasattr(self.vqvae.encoder, 'conv_in'):
            dtype = self.vqvae.encoder.conv_in.weight.dtype
        else:
            # 尝试获取任何参数的数据类型
            for param in self.vqvae.parameters():
                dtype = param.dtype
                break
        
        if dtype is not None:
            latents = latents.to(dtype)
            
        # 解码
        try:
            image = self.vqvae.decode(latents, return_dict=False)[0]
            
            # 规范化到[0, 1]范围
            image = (image / 2 + 0.5).clamp(0, 1)
            
            # 转换为numpy数组，形状为[B, H, W, C]
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            
            return image
        except Exception as e:
            print(f"VAE解码失败: {e}")
            # 创建一个空的占位图像
            print("创建占位图像")
            placeholder = np.zeros((latents.shape[0], 256, 256, 3), dtype=np.float32)
            return placeholder

    def enable_vae_slicing(self):
        """
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow for larger batch sizes.
        """
        self.vqvae.enable_slicing()

    def disable_vae_slicing(self):
        """
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vqvae.disable_slicing()

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            user_ids: Optional[torch.LongTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            guidance_scale: float = 7.5,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        
        # 0. 获取设备和数据类型
        device = self.device
        dtype = self.unet.dtype

        # 1. 定义调用默认值
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. 准备时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. 准备潜在变量
        # 修复：确保使用正确的通道数（与训练时一致）
        num_channels_latents = 3  # 与UNet配置保持一致
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 4. 准备额外的调度器参数
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5. 无分类器指导设置
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if user_ids is None:
                raise ValueError("`user_ids`不能为空，当 `guidance_scale` > 1")
            
            # 创建无条件标签
            uncond_ids = torch.full_like(user_ids, self.unet.config.num_class_embeds - 1)
            class_labels = torch.cat([user_ids, uncond_ids])
        else:
            class_labels = user_ids

        # 6. 去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 如果使用无分类器指导，扩展潜在变量
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 预测噪声残差
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    class_labels=class_labels,
                    return_dict=False,
                )[0]

                # 执行指导
                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # 计算上一步的样本
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # 更新进度条
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. 将潜在变量解码为图像
        image = self.decode_latents(latents)

        # 9. 转换为PIL Image
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image) 