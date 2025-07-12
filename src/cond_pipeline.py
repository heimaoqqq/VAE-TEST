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
from src.cond_unet import CondUNet2DModel


class CondLatentDiffusionPipeline(LatentDiffusionPipelineBase):
    """
    条件潜在扩散Pipeline，支持用户ID控制
    """
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
            unet: CondUNet2DModel,
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
            batch_size: int = 1,
            user_ids: Optional[torch.Tensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            guidance_scale: float = 1.0,  # 添加条件引导比例
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        """
        条件生成函数
        
        Args:
            batch_size: 批次大小
            user_ids: 用户ID张量 [batch_size]
            height: 输出图像高度
            width: 输出图像宽度
            num_inference_steps: 推理步数
            generator: 随机数生成器
            latents: 预定义的潜在向量
            output_type: 输出类型 "pil"或"numpy"
            return_dict: 是否返回字典
            eta: DDIM采样器的eta参数
            guidance_scale: 条件引导比例
        """
        # 检查GPU数量和配置
        gpu_count = torch.cuda.device_count()
        is_dataparallel = isinstance(self.unet.unet, nn.DataParallel)  # 注意这里访问了unet.unet
        
        print(f"可用GPU数量: {gpu_count}")
        if gpu_count > 1:
            print(f"使用 {gpu_count} 个GPU进行并行推理")
            if not is_dataparallel:
                print("配置DataParallel...")
                self.unet.unet = nn.DataParallel(self.unet.unet)  # 只并行内部的UNet
                is_dataparallel = True
        
        # 确定正确的设备
        if is_dataparallel:
            device = next(self.unet.unet.module.parameters()).device  # 注意这里的路径变化
            unet_config = self.unet.unet.module.config  # 注意这里的路径变化
            unet_dtype = next(self.unet.unet.module.parameters()).dtype
        else:
            device = self.device
            unet_config = self.unet.unet.config  # 注意这里的路径变化
            unet_dtype = self.unet.dtype
            
        # 处理user_ids，确保它在正确的设备上
        if user_ids is not None:
            if not torch.is_tensor(user_ids):
                user_ids = torch.tensor([user_ids], device=device)
            elif user_ids.device != device:
                user_ids = user_ids.to(device)
            
            # 确保user_ids长度与batch_size一致
            if len(user_ids) == 1 and batch_size > 1:
                user_ids = user_ids.repeat(batch_size)
            elif len(user_ids) != batch_size:
                raise ValueError(f"user_ids长度({len(user_ids)})与batch_size({batch_size})不一致")
                
        # 设置图像大小
        height = height or unet_config.sample_size * self.vae_scale_factor
        width = width or unet_config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height`和`width`必须是8的倍数，当前值为{height}和{width}。"
            )

        # 根据GPU数量确定每个GPU的批量大小
        if gpu_count > 1:
            original_batch_size = batch_size
            adjusted_batch_size = batch_size
            if batch_size % gpu_count != 0:
                adjusted_batch_size = ((batch_size // gpu_count) + 1) * gpu_count
                print(f"批量大小调整为: {adjusted_batch_size} (GPU数量的倍数)")
                if adjusted_batch_size != batch_size:
                    batch_size = adjusted_batch_size
                    # 如果有user_ids，也需要扩展
                    if user_ids is not None:
                        # 复制最后一个用户ID来填充
                        padding = user_ids[-1].repeat(adjusted_batch_size - original_batch_size)
                        user_ids = torch.cat([user_ids, padding])
        else:
            original_batch_size = batch_size
        
        # 准备latents
        shape = (batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"generator列表长度({len(generator)})与batch_size({batch_size})不一致"
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=unet_dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"latents形状不匹配，期望{shape}，实际{latents.shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 使用tqdm显示进度
        for t in tqdm(self.scheduler.timesteps, desc="生成进度"):
            # 确保时间步是正确的形状
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 当前latents
            current_latents = latents
            
            # 对输入进行缩放
            latents = self.scheduler.scale_model_input(current_latents, t)
            
            # 条件生成
            if guidance_scale > 1.0 and user_ids is not None:
                # 运行无条件前向传播
                with torch.no_grad():
                    noise_pred_uncond = self.unet(latents, t_tensor, user_ids=None).sample
                    
                # 运行条件前向传播
                noise_pred_cond = self.unet(latents, t_tensor, user_ids=user_ids).sample
                
                # 进行引导组合
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 直接运行前向传播
                noise_pred = self.unet(latents, t_tensor, user_ids=user_ids).sample
            
            # 进行去噪步骤
            latents = self.scheduler.step(
                noise_pred, t, current_latents, **extra_step_kwargs
            ).prev_sample

        # 仅保留请求的图像数量
        if batch_size > original_batch_size:
            latents = latents[:original_batch_size]
            
        # 解码潜在表示得到图像
        image = self.decode_latents(latents)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image) 