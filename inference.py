# from src.pipeline import UncondLatentDiffusionPipeline
# import torch
#
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#
# model_id = "path to your model here"
#
# # load model
# pipeline = UncondLatentDiffusionPipeline.from_pretrained(model_id).to(device)
#
# image = pipeline(num_inference_steps=1000).images[0]
#
# # Save the image
# image.save("generated_image.png")

from src.pipeline import UncondLatentDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "path to your model folder here"

# load model
pipeline = UncondLatentDiffusionPipeline.from_pretrained(model_id).to(device)

image = pipeline(num_inference_steps=1000).images[0]

# Save the image
image.save("generated_image.png")
