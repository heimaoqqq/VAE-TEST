import inspect
import os
from typing import Callable, List, Optional, Union, Tuple
import math
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    VQModel,
    UNet2DModel,
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


class LatentDiffusionPipelineBase(DiffusionPipeline):
    """
    潜在扩散Pipeline基类
    """
    
    @property
    def device(self):
        """获取设备属性"""
        # 首先检查是否有_device属性
        if hasattr(self, '_device'):
            return self._device
            
        # 尝试从组件获取设备
        for component in [self.unet, self.vae]:
            if hasattr(component, 'device'):
                return component.device
                
        # 回退到第一个参数的设备
        for component in self.components.values():
            if hasattr(component, 'parameters'):
                try:
                    return next(component.parameters()).device
                except StopIteration:
                    continue
                    
        # 最终回退
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def decode_latents(self, latents):
        latents = latents / 0.18215
        # 确保latents与vae的数据类型一致
        dtype = self.vae.dtype if hasattr(self.vae, 'dtype') else self.vae.encoder.conv_in.weight.dtype
        latents = latents.to(dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


class UncondLatentDiffusionPipeline(LatentDiffusionPipelineBase):
    def __init__(
            self,
            vae: VQModel,
            scheduler: Union[
                DDIMScheduler,
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
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,  # default to generate a single image
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        # 检查GPU数量和配置
        gpu_count = torch.cuda.device_count()
        is_dataparallel = isinstance(self.unet, nn.DataParallel)
        
        print(f"可用GPU数量: {gpu_count}")
        if gpu_count > 1:
            print(f"使用 {gpu_count} 个GPU进行并行推理")
            if not is_dataparallel:
                print("配置DataParallel...")
                self.unet = nn.DataParallel(self.unet)
                is_dataparallel = True
        
        # 确定正确的设备
        if is_dataparallel:
            device = next(self.unet.module.parameters()).device
            unet_config = self.unet.module.config
            unet_dtype = next(self.unet.module.parameters()).dtype
        else:
            device = self.device
            unet_config = self.unet.config
            unet_dtype = self.unet.dtype
            
        # 0. Default height and width to unet
        height = height or unet_config.sample_size * self.vae_scale_factor
        width = width or unet_config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # 根据GPU数量确定每个GPU的批量大小
        if gpu_count > 1:
            # 保存原始请求的批量大小
            original_batch_size = batch_size
            # 调整为GPU数量的倍数
            adjusted_batch_size = batch_size
            if batch_size % gpu_count != 0:
                adjusted_batch_size = ((batch_size // gpu_count) + 1) * gpu_count
                print(f"批量大小调整为: {adjusted_batch_size} (GPU数量的倍数)")
                if adjusted_batch_size != batch_size:
                    # 处理所有请求的图像，但可能会多生成一些
                    batch_size = adjusted_batch_size
        else:
            original_batch_size = batch_size
        
        # 准备latents
        shape = (batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=unet_dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 使用tqdm显示进度
        for t in tqdm(self.scheduler.timesteps, desc="生成进度"):
            # 确保时间步是正确的形状，以适应多GPU
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 扩展输入以匹配批量大小
            current_latents = latents
            
            # 对输入进行模型推理
            latents = self.scheduler.scale_model_input(current_latents, t)
            
            # 使用UNet预测噪声
            if is_dataparallel:
                try:
                    # 分离模块进行直接调用，避免DataParallel问题
                    unet_module = self.unet.module
                    noise_pred = unet_module(latents, t_tensor).sample
                except Exception as e:
                    print(f"使用module直接调用失败: {e}")
                    print("回退到分批处理...")
                    
                    # 分批处理以避免DataParallel问题
                    chunks = gpu_count
                    chunk_size = batch_size // chunks
                    noise_preds = []
                    
                    for i in range(chunks):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size
                        
                        # 处理每个块
                        chunk_latents = latents[start_idx:end_idx]
                        chunk_t = t_tensor[start_idx:end_idx]
                        
                        # 使用module直接处理
                        with torch.no_grad():
                            chunk_output = unet_module(chunk_latents, chunk_t).sample
                            noise_preds.append(chunk_output)
                    
                    # 合并结果
                    noise_pred = torch.cat(noise_preds, dim=0)
            else:
                noise_pred = self.unet(latents, t).sample
            
            # 进行去噪步骤
            latents = self.scheduler.step(
                noise_pred, t, current_latents, **extra_step_kwargs
            ).prev_sample

        # 仅保留请求的图像数量
        if batch_size > original_batch_size:
            latents = latents[:original_batch_size]
            
        # scale and decode the image latents with vae
        image = self.decode_latents(latents)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
