## Training an unconditional latent diffusion model

Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

### Cloning to local
```bash
git clone https://github.com/zyinghua/uncond-image-generation-ldm.git
```

Then call:
```bash
cd uncond-image-generation-ldm
```

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Change Pretrained VAE settings
You can specify which pretrained VAE model to use by changing the `VAE_PRETRAINED_PATH` and `VAE_KWARGS` variables in `train.py`, at the top.

### Unconditional Flowers

An examplar command to train a DDPM UNet model on the Oxford Flowers dataset, without using GPUs:

```bash
accelerate launch train.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=256 \
  --output_dir="ddpm-ema-flowers-256" \
  --train_batch_size=16 \
  --num_epochs=150 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
```

### Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. After setting up with `accelerate config`,
simply add `--multi_gpu` in the command. For more information, follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash
accelerate launch --multi_gpu train.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=256 \
  --output_dir="ddpm-ema-flowers-256" \
  --train_batch_size=16 \
  --num_epochs=150 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
```

To be able to use Weights and Biases (`wandb`) as a logger you need to install the library: `pip install wandb`.

### Using your own data

To use your own dataset, there are 3 ways:
- you can either provide your own folder as `--train_data_dir`
- or you can provide your own .zip file containing the data as `--train_data_files`
- or you can upload your dataset to the hub (possibly as a private repo, if you prefer so), and simply pass the `--dataset_name` argument.

Below, we explain both in more detail.

#### Provide the dataset as a folder/zip file

If you provide your own folders with images, the script expects the following directory structure:

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

In other words, the script will take care of gathering all images inside the folder. You can then run the script like this:

```bash
accelerate launch train.py \
    --train_data_dir <path-to-train-directory> \
    <other-arguments>
```

Or (if it is a zip file):
```bash
accelerate launch train.py \
    --train_data_files <path-to-train-zip-file> \
    <other-arguments>
```

Internally, the script will use the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature which will automatically turn the folders into 🤗 Dataset objects.

Official [diffusers](https://github.com/huggingface/diffusers) repo also has a pipeline for uncond ldm that can be found [here](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/deprecated/latent_diffusion_uncond).

