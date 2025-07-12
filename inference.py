from src.cond_pipeline import CondLatentDiffusionPipeline
from diffusers import UNet2DModel
import torch
import os
from PIL import Image
import numpy as np
import time
import threading
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser(description='条件微多普勒时频图生成')
parser.add_argument('--model_path', type=str, default="/kaggle/working/VAE",
                    help='模型路径')
parser.add_argument('--user_id', type=int, default=1,
                    help='要生成的用户ID（1-31）对应文件夹ID_1到ID_31')
parser.add_argument('--batch_size', type=int, default=8,
                    help='批量生成数量')
parser.add_argument('--steps', type=int, default=1000,
                    help='扩散步数')
parser.add_argument('--guidance_scale', type=float, default=7.5,
                    help='条件引导强度，越高生成越符合条件，但多样性降低（推荐值：7.5-15.0）')
parser.add_argument('--output_dir', type=str, default="generated_images",
                    help='输出目录')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子')
parser.add_argument('--validate_conditions', action='store_true',
                    help='开启条件验证模式，为多个用户ID生成相同种子的图像以比较效果')
args = parser.parse_args()

# 检测可用GPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
gpu_count = torch.cuda.device_count()
print(f"检测到 {gpu_count} 个GPU设备")

# 设置环境变量以启用多GPU
if gpu_count > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(gpu_count)])
    print(f"设置环境变量CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

model_id = args.model_path

# 检查模型路径是否存在
if not os.path.exists(model_id):
    print(f"错误：模型路径 {model_id} 不存在！")
    print("请确保已经训练并保存了模型，或者修改model_id为正确的路径。")
    exit(1)

# 检查用户ID是否在有效范围内
if args.user_id < 1 or args.user_id > 31:
    print(f"错误：用户ID必须在1到31之间，当前值为{args.user_id}")
    exit(1)

# 将文件夹ID转换为模型内部ID（从0开始）
model_user_id = args.user_id - 1
print(f"生成用户ID_{args.user_id}的图像（模型内部ID: {model_user_id}）")

try:
    print(f"正在从 {model_id} 加载模型组件...")
    start_time = time.time()
    
    # 手动加载各个组件并检查不同可能的路径
    from diffusers import DDPMScheduler, VQModel
    
    print("加载VAE模型...")
    vae = VQModel.from_pretrained(os.path.join(model_id, "vae"))
    
    print("加载调度器...")
    scheduler = DDPMScheduler.from_pretrained(os.path.join(model_id, "scheduler"))
    
    print("加载条件UNet模型...")
    # 直接使用UNet2DModel的from_pretrained方法加载整个UNet
    unet = UNet2DModel.from_pretrained(os.path.join(model_id, "unet"))
    
    # 创建条件Pipeline
    print("创建条件Pipeline...")
    pipeline = CondLatentDiffusionPipeline(
        vae=vae,
        scheduler=scheduler,
        unet=unet
    )

    # 移动模型到设备
    print(f"将模型移动到设备 {device}...")
    vae = vae.to(device)
    unet = unet.to(device)
    pipeline = pipeline.to(device)
    
    load_time = time.time() - start_time
    print(f"模型加载成功！耗时 {load_time:.2f} 秒")
except Exception as e:
    print(f"加载模型时出错: {e}")
    import traceback
    traceback.print_exc()
    print("请检查模型路径和文件是否完整。")
    exit(1)

# 如果有多个GPU，采用手动并行策略
if gpu_count > 1:
    try:
        # 尝试创建第二个模型实例到第二个GPU
        print(f"正在为第二个GPU加载模型...")
        second_device = f"cuda:1"
        
        # 为第二个GPU加载条件UNet
        print("为第二个GPU加载条件UNet模型...")
        unet2 = UNet2DModel.from_pretrained(os.path.join(model_id, "unet"))
        
        print("为第二个GPU创建Pipeline...")
        pipeline2 = CondLatentDiffusionPipeline(
            vae=VQModel.from_pretrained(os.path.join(model_id, "vae")),  # 创建新的VAE实例
            scheduler=scheduler,  # 调度器不需要移动到GPU
            unet=unet2
        )
        
        # 将组件移动到第二个GPU
        print(f"将模型移动到设备 {second_device}...")
        pipeline2.vae = pipeline2.vae.to(second_device)
        pipeline2.unet = pipeline2.unet.to(second_device)
        pipeline2 = pipeline2.to(second_device)
        
        print(f"第二个模型实例加载成功！")
        use_dual_pipeline = True
    except Exception as e:
        print(f"无法加载第二个模型实例: {e}")
        import traceback
        traceback.print_exc()
        print("将仅使用单个GPU")
        use_dual_pipeline = False
else:
    use_dual_pipeline = False

# 生成参数
if use_dual_pipeline:
    batch_size = args.batch_size  # 总批量大小
    per_gpu_batch = batch_size // 2  # 每个GPU处理一半
    print(f"使用双GPU并行，每个GPU处理 {per_gpu_batch} 个样本")
else:
    batch_size = args.batch_size
    print(f"使用单GPU，批量大小为 {batch_size}")

num_inference_steps = args.steps  # 推理步数
output_dir = args.output_dir  # 输出目录
guidance_scale = args.guidance_scale  # 条件引导强度

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子以确保可重复性
seed = args.seed
print(f"使用随机种子 {seed}")
generator1 = torch.Generator(device="cuda:0").manual_seed(seed)
if use_dual_pipeline:
    generator2 = torch.Generator(device="cuda:1").manual_seed(seed+1000)

# 双GPU并行生成函数
def generate_on_gpu(pipeline, batch_size, num_steps, user_id, generator, guidance_scale, device_idx, results):
    print(f"GPU {device_idx} 开始生成用户ID_{args.user_id}的 {batch_size} 张图像...")
    start_time = time.time()
    # 创建用户ID张量
    user_ids = torch.tensor([user_id] * batch_size, device=f"cuda:{device_idx}")
    
    images = pipeline(
        batch_size=batch_size,
        num_inference_steps=num_steps,
        generator=generator,
        user_ids=user_ids,
        guidance_scale=guidance_scale  # 添加条件引导强度
    ).images
    generation_time = time.time() - start_time
    print(f"GPU {device_idx} 完成！耗时 {generation_time:.2f} 秒")
    results[device_idx] = images

# 生成微多普勒时频图像
print(f"开始生成用户ID_{args.user_id}的 {batch_size} 张图像，使用 {num_inference_steps} 步推理...")
start_time = time.time()

if use_dual_pipeline:
    # 使用两个独立的GPU进行并行处理
    results = [None, None]
    
    # 创建线程
    t1 = threading.Thread(target=generate_on_gpu, 
                          args=(pipeline, per_gpu_batch, num_inference_steps, model_user_id, generator1, guidance_scale, 0, results))
    t2 = threading.Thread(target=generate_on_gpu, 
                          args=(pipeline2, per_gpu_batch, num_inference_steps, model_user_id, generator2, guidance_scale, 1, results))
    
    # 启动线程
    t1.start()
    t2.start()
    
    # 等待两个线程完成
    t1.join()
    t2.join()
    
    # 合并结果
    images = []
    for result in results:
        if result is not None:
            images.extend(result)
else:
    # 单GPU处理
    # 创建用户ID张量
    user_ids = torch.tensor([model_user_id] * batch_size, device=device)
    
    images = pipeline(
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        generator=generator1,
        user_ids=user_ids,
        guidance_scale=guidance_scale  # 添加条件引导强度
    ).images

generation_time = time.time() - start_time
print(f"图像生成完成！总耗时 {generation_time:.2f} 秒，平均每张 {generation_time/batch_size:.2f} 秒")

# 保存生成的图像
for i, image in enumerate(images):
    image.save(os.path.join(output_dir, f"ID_{args.user_id}_microdoppler_{i:03d}.png"))

print(f"成功生成 {len(images)} 张用户ID_{args.user_id}的微多普勒时频图像，保存在 {output_dir} 目录")

# 可选：创建一个包含所有图像的网格展示图
def create_image_grid(images, rows, cols):
    if not images:
        print("警告：没有生成图像，跳过网格创建")
        return None
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

# 创建网格图像
if images:
    rows = int(np.sqrt(len(images)))
    if rows == 0:  # 防止除零错误
        rows = 1
    cols = len(images) // rows + (1 if len(images) % rows != 0 else 0)
    grid = create_image_grid(images, rows, cols)
    if grid:
        grid.save(os.path.join(output_dir, f"ID_{args.user_id}_microdoppler_grid.png"))
        print(f"创建网格展示图：{os.path.join(output_dir, f'ID_{args.user_id}_microdoppler_grid.png')}")
else:
    print("没有生成任何图像，跳过创建网格图像")

# 性能统计
if use_dual_pipeline:
    print("\n性能统计:")
    print(f"GPU数量: 2 (手动并行)")
    print(f"总批量大小: {batch_size}")
    print(f"每个GPU的批量大小: {per_gpu_batch}")
    print(f"总生成时间: {generation_time:.2f}秒")
    print(f"平均每张图像时间: {generation_time/batch_size:.2f}秒")
    print(f"每GPU每秒生成图像数: {batch_size / generation_time / 2:.2f}")

# 条件验证模式：为多个用户ID生成相同种子的图像
if args.validate_conditions:
    print("\n开始条件验证模式：为多个用户ID生成相同种子的图像以比较效果...")
    
    # 选择一些用户ID进行比较（比如1、5、10、15、20）
    compare_user_ids = [1, 5, 10, 15, 20]
    compare_output_dir = os.path.join(output_dir, "condition_comparison")
    os.makedirs(compare_output_dir, exist_ok=True)
    
    # 使用较小的批次大小和更少的步数以加快验证过程
    validation_batch_size = 4
    validation_steps = min(args.steps, 500)
    
    # 使用固定种子，确保可比性
    validation_seed = 1234
    validation_generator = torch.Generator(device=device).manual_seed(validation_seed)
    
    # 创建比较网格的图像列表
    comparison_images = []
    
    for user_id in compare_user_ids:
        print(f"\n生成用户ID_{user_id}的验证图像...")
        model_user_id = user_id - 1  # 转换为模型内部ID
        user_ids = torch.tensor([model_user_id] * validation_batch_size, device=device)
        
        # 生成图像
        images = pipeline(
            batch_size=validation_batch_size,
            num_inference_steps=validation_steps,
            generator=validation_generator,
            user_ids=user_ids,
            guidance_scale=args.guidance_scale
        ).images
        
        # 保存第一张图像用于比较
        if images:
            comparison_images.append(images[0])
            images[0].save(os.path.join(compare_output_dir, f"ID_{user_id}_comparison.png"))
    
    # 创建比较网格
    if len(comparison_images) > 1:
        comparison_grid = create_image_grid(comparison_images, 1, len(comparison_images))
        comparison_grid.save(os.path.join(compare_output_dir, f"user_id_comparison_grid.png"))
        print(f"\n创建用户ID比较网格图像: {os.path.join(compare_output_dir, 'user_id_comparison_grid.png')}")
        print("请查看比较图像以评估条件控制效果")
