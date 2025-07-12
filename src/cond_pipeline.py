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
        
        # 设置内部设备属性，基类的device属性会使用这个
        if hasattr(unet, 'device'):
            self._device = unet.device
        elif hasattr(vae, 'device'):
            self._device = vae.device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
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
        device = self.device
        
        # 安全检测模型类型，处理不同的并行包装类型
        is_dataparallel = False
        
        # 检查unet是否已经是DataParallel
        if isinstance(self.unet, nn.DataParallel) or isinstance(self.unet, nn.parallel.DistributedDataParallel):
            is_dataparallel = True
            print("检测到UNet已经是DataParallel模型")
        elif hasattr(self.unet, 'unet'):
            # 检查内部unet是否为DataParallel
            if isinstance(self.unet.unet, nn.DataParallel) or isinstance(self.unet.unet, nn.parallel.DistributedDataParallel):
                is_dataparallel = True
                print("检测到内部UNet已经是DataParallel模型")
        
        print(f"可用GPU数量: {gpu_count}")
        if gpu_count > 1 and not is_dataparallel:
            print(f"使用 {gpu_count} 个GPU进行并行推理")
            # 我们不再尝试在这里包装模型，因为我们已经在CondUNet2DModel中处理了DataParallel的情况
        
        # 确定正确的设备
        unet_config = None
        unet_dtype = None
        
        # 获取UNet配置的安全方法
        def get_unet_config_safe(unet_model):
            try:
                # 尝试直接获取配置
                if hasattr(unet_model, 'config'):
                    return unet_model.config, next(unet_model.parameters()).dtype
                # 尝试通过module获取配置
                elif hasattr(unet_model, 'module') and hasattr(unet_model.module, 'config'):
                    return unet_model.module.config, next(unet_model.module.parameters()).dtype
                # 尝试通过unet属性获取配置
                elif hasattr(unet_model, 'unet'):
                    inner_unet = unet_model.unet
                    if hasattr(inner_unet, 'config'):
                        return inner_unet.config, next(inner_unet.parameters()).dtype
                    elif hasattr(inner_unet, 'module') and hasattr(inner_unet.module, 'config'):
                        return inner_unet.module.config, next(inner_unet.module.parameters()).dtype
                return None, torch.float32
            except Exception as e:
                print(f"获取UNet配置失败: {e}")
                return None, torch.float32
        
        # 获取UNet配置
        unet_config, unet_dtype = get_unet_config_safe(self.unet)
        if unet_config is None:
            print("警告: 无法获取UNet配置，使用默认值")
            # 尝试使用默认的sample_size
            if hasattr(self.unet, 'config'):
                unet_config = self.unet.config
            else:
                # 创建一个简单的配置对象
                class SimpleConfig:
                    def __init__(self, sample_size=32):
                        self.sample_size = sample_size
                unet_config = SimpleConfig()
            
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
                # 运行无条件前向传播 - 使用-1作为无条件标记
                uncond_ids = torch.full_like(user_ids, -1)
                
                # 直接调用模型，让模型内部处理DataParallel的情况
                noise_pred_uncond = self.unet(latents, t_tensor, user_ids=uncond_ids).sample
                
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
