import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL, VQModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from src.pipeline import *


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# Change the info of your pretrained VAE model here
VAE_PRETRAINED_PATH = "CompVis/ldm-celebahq-256"  # 使用预训练的256x256 VAE模型
VAE_KWARGS = {"subfolder":"vqvae"}  # 保持默认参数


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def preprocess_microdoppler_data(data_dir, output_dir, image_size=256, max_samples=None):
    """
    预处理微多普勒时频图数据，确保所有图像尺寸一致且为RGB格式
    
    Args:
        data_dir: 原始数据目录
        output_dir: 处理后数据保存目录
        image_size: 目标图像尺寸
        max_samples: 最大样本数量，None表示处理所有样本
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        logger.error(f"数据目录 {data_dir} 不存在")
        return
        
    # 遍历所有用户目录
    try:
        user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not user_dirs:
            logger.warning(f"在 {data_dir} 中未找到子目录，尝试处理根目录中的图像")
            # 如果没有子目录，则将根目录视为单个用户目录
            user_dirs = ["default"]
            
        processed_count = 0
        
        for user_dir in tqdm(user_dirs, desc="处理用户数据"):
            # 处理默认情况
            if user_dir == "default":
                user_path = data_dir
            else:
                user_path = os.path.join(data_dir, user_dir)
                
            output_user_path = os.path.join(output_dir, user_dir)
            os.makedirs(output_user_path, exist_ok=True)
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.warning(f"在 {user_path} 中未找到图像文件")
                continue
                
            for img_file in image_files:
                if max_samples is not None and processed_count >= max_samples:
                    logger.info(f"已达到最大样本数 {max_samples}，停止处理")
                    return
                
                img_path = os.path.join(user_path, img_file)
                output_img_path = os.path.join(output_user_path, img_file)
                
                # 如果输出文件已存在，则跳过
                if os.path.exists(output_img_path):
                    logger.debug(f"文件 {output_img_path} 已存在，跳过处理")
                    processed_count += 1
                    continue
                    
                try:
                    # 读取图像
                    img = Image.open(img_path)
                    
                    # 确保图像是RGB格式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 应用转换
                    processed_img = transform(img)
                    
                    # 转回PIL图像并保存
                    processed_img = ((processed_img + 1.0) * 127.5).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    processed_img = Image.fromarray(processed_img)
                    processed_img.save(output_img_path)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"处理图像 {img_path} 时出错: {e}")
        
        logger.info(f"成功处理 {processed_count} 张微多普勒时频图像")
    except Exception as e:
        logger.error(f"预处理过程中发生错误: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="只进行数据预处理，不训练模型",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="原始微多普勒时频图数据目录，用于预处理",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=None,
        help="预处理后的数据保存目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="预处理时的最大样本数量，None表示处理所有样本",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/kaggle/working/dataset",  # 设置默认值为dataset目录
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-256",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=20, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2500,  # 从500修改为2500，减少检查点保存频率
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,  # 修改默认值为1，只保留最新的检查点
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--acc_seed",
        type=int,
        default=None,
        help="A seed to reproduce the training. If not set, the seed will be random.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        default=None,
        help=(
            "The files of the training data. The files must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    # 确保shutil模块在所有作用域可用
    import shutil
    
    # 处理微多普勒数据（如果需要）
    if args.raw_data_dir is not None and args.processed_data_dir is not None:
        logger.info("检测到原始数据目录和处理后目录，执行数据预处理...")
        preprocess_microdoppler_data(args.raw_data_dir, args.processed_data_dir, args.resolution, args.max_samples)
        # 将处理后的数据目录设置为训练数据目录
        args.train_data_dir = args.processed_data_dir
        logger.info(f"预处理完成，将使用处理后的数据：{args.processed_data_dir}")
    
    # preprocess_only模式下不应该执行到这里，但为了安全起见
    if args.preprocess_only:
        logger.info("preprocess_only模式，跳过训练")
        return
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Set the random seed manually for reproducibility.
    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load pretrained VAE model
    vae = VQModel.from_pretrained(VAE_PRETRAINED_PATH, **VAE_KWARGS)

    # Freeze the VAE model
    vae.requires_grad_(False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(sample_size=args.resolution // vae_scale_factor,
                            in_channels=3, out_channels=3)
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 加载数据集
    if args.dataset_name is not None:
        # 从Hugging Face Hub下载数据集
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        # 从本地加载数据集
        try:
            # 打印数据目录信息
            logger.info(f"尝试从 {args.train_data_dir} 加载数据集")
            if os.path.exists(args.train_data_dir):
                logger.info(f"目录存在，内容: {os.listdir(args.train_data_dir)}")
                
                # 检查是否已有ID_x目录结构
                has_id_dirs = any(d.startswith("ID_") for d in os.listdir(args.train_data_dir))
                if has_id_dirs and accelerator.is_main_process:  # 只让主进程执行目录创建
                    logger.info("检测到ID_x目录结构，创建符合imagefolder格式的目录...")
                    # 创建train目录
                    train_dir = os.path.join(args.train_data_dir, "train")
                    os.makedirs(train_dir, exist_ok=True)
                    
                    # 将ID_x目录移动到train目录下
                    for item in os.listdir(args.train_data_dir):
                        if item.startswith("ID_") and os.path.isdir(os.path.join(args.train_data_dir, item)):
                            src_dir = os.path.join(args.train_data_dir, item)
                            dst_dir = os.path.join(train_dir, item)
                            try:
                                if not os.path.exists(dst_dir):
                                    logger.info(f"复制目录 {item} 到 train/{item}")
                                    # 使用copy_tree而不是copytree，手动实现目录复制以避免目录已存在错误
                                    os.makedirs(dst_dir, exist_ok=True)
                                    for file_name in os.listdir(src_dir):
                                        src_file = os.path.join(src_dir, file_name)
                                        dst_file = os.path.join(dst_dir, file_name)
                                        if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                            shutil.copy2(src_file, dst_file)
                                else:
                                    logger.info(f"目录 train/{item} 已存在，检查文件是否需要复制")
                                    # 检查并复制缺失的文件
                                    for file_name in os.listdir(src_dir):
                                        src_file = os.path.join(src_dir, file_name)
                                        dst_file = os.path.join(dst_dir, file_name)
                                        if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                            shutil.copy2(src_file, dst_file)
                            except Exception as e:
                                logger.warning(f"复制目录 {item} 时出错: {e}")
                                # 继续处理其他目录
                    
                    logger.info(f"train目录结构创建完成，内容: {os.listdir(train_dir)}")
                
                # 确保所有进程等待主进程完成目录创建
                accelerator.wait_for_everyone()
            else:
                logger.info(f"目录不存在，将创建")
                os.makedirs(args.train_data_dir, exist_ok=True)
                
            # 尝试标准的imagefolder格式加载
            dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        except ValueError as e:
            if "corresponds to no data" in str(e):
                logger.info("标准imagefolder格式加载失败，尝试直接加载图像...")
                # 创建正确的目录结构
                # 确保shutil可用
                import shutil  # 重新导入以确保在此作用域可用
                
                # 只让主进程执行目录创建
                if accelerator.is_main_process:
                    train_dir = os.path.join(args.train_data_dir, "train")
                    os.makedirs(train_dir, exist_ok=True)
                    default_class_dir = os.path.join(train_dir, "default_class")
                    os.makedirs(default_class_dir, exist_ok=True)
                    
                    # 移动所有图像到默认类别目录
                    image_count = 0
                    for file in os.listdir(args.train_data_dir):
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            src_path = os.path.join(args.train_data_dir, file)
                            dst_path = os.path.join(default_class_dir, file)
                            if not os.path.exists(dst_path):  # 避免重复复制
                                shutil.copy(src_path, dst_path)  # 使用复制而不是移动，以保留原始文件
                                image_count += 1
                    
                    logger.info(f"已复制 {image_count} 张图像到标准目录结构")
                
                # 确保所有进程等待主进程完成目录创建
                accelerator.wait_for_everyone()
                
                # 重新尝试加载数据集
                dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
            else:
                # 其他错误，直接抛出
                raise

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler, vae = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, vae
    )

    # 设置VAE模型的数据类型，但保留设备分配
    if accelerator.mixed_precision == "fp16":
        vae.to(dtype=torch.float16)
    elif accelerator.mixed_precision == "bf16":
        vae.to(dtype=torch.bfloat16)

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"].to(weight_dtype)
            latents = vae.encode(clean_images).latents
            
            # 确保latents与weight_dtype一致
            latents = latents.to(dtype=weight_dtype)
            
            latents = latents * 0.18215#vae.config.scaling_factor

            # Sample noise that we'll add to the images
            noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
            bsz = latents.shape[0]  # batch size
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_latents, timesteps).sample

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # 强制清理所有现有检查点，确保只保留最新的
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        if len(checkpoints) > 0:
                            logger.info(f"清理所有现有检查点，共 {len(checkpoints)} 个")
                            for checkpoint in checkpoints:
                                checkpoint_path = os.path.join(args.output_dir, checkpoint)
                                logger.info(f"删除检查点: {checkpoint}")
                                shutil.rmtree(checkpoint_path)
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"保存新检查点到 {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = UncondLatentDiffusionPipeline(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                # 清理旧的完整模型文件，减少存储空间占用
                model_files = ["model_index.json", "scheduler", "unet"]
                if args.use_ema:
                    model_files.append("unet_ema")
                
                logger.info("清理旧的完整模型文件...")
                for file_or_dir in model_files:
                    path = os.path.join(args.output_dir, file_or_dir)
                    if os.path.isdir(path):
                        logger.info(f"删除目录: {file_or_dir}")
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.isfile(path):
                        logger.info(f"删除文件: {file_or_dir}")
                        os.remove(path)

                pipeline = UncondLatentDiffusionPipeline(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)
                logger.info(f"保存完整模型到 {args.output_dir}")

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    
    # 检查是否只执行预处理
    if args.preprocess_only:
        if args.raw_data_dir is None or args.processed_data_dir is None:
            raise ValueError("When --preprocess_only is True, --raw_data_dir and --processed_data_dir must be specified.")
        # 使用标准日志而非accelerate日志
        import logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logging.info("仅执行数据预处理...")
        preprocess_microdoppler_data(args.raw_data_dir, args.processed_data_dir, args.resolution, args.max_samples)
        logging.info("预处理完成，退出程序")
        exit(0)  # 添加退出语句，确保不继续执行
    else:
        # 检查必要的参数
        if args.dataset_name is None and args.train_data_files is None and args.train_data_dir is None:
            raise ValueError("You must specify either a dataset name from the hub or a train data directory.")
        main(args)
