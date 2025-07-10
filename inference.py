from src.pipeline import UncondLatentDiffusionPipeline
import torch
import os
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "path/to/your/model"  # 修改为您的模型路径

# 加载模型
pipeline = UncondLatentDiffusionPipeline.from_pretrained(model_id).to(device)

# 生成参数
batch_size = 16  # 一次生成的图像数量
num_inference_steps = 1000  # 推理步数
output_dir = "generated_images"  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 生成微多普勒时频图像
images = pipeline(
    batch_size=batch_size,
    num_inference_steps=num_inference_steps
).images

# 保存生成的图像
for i, image in enumerate(images):
    image.save(os.path.join(output_dir, f"microdoppler_spectrogram_{i:03d}.png"))

print(f"成功生成 {batch_size} 张微多普勒时频图像，保存在 {output_dir} 目录")

# 可选：创建一个包含所有图像的网格展示图
def create_image_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

# 创建网格图像
rows = int(np.sqrt(batch_size))
cols = batch_size // rows + (1 if batch_size % rows != 0 else 0)
grid = create_image_grid(images[:batch_size], rows, cols)
grid.save(os.path.join(output_dir, "microdoppler_grid.png"))
print(f"创建网格展示图：{os.path.join(output_dir, 'microdoppler_grid.png')}")
