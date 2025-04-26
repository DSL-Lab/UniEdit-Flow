
<div align="center">

<h1> 🤗 UniEdit-Flow_diffusers </h1>

</div>

> The implementation of UniEdit-Flow based on diffusers.

<h1> 🛠️ Environment </h1>

We provide an environment setup based on [Anaconda](https://www.anaconda.com/):

```shell
conda create -n unieditflow python==3.10
conda activate unieditflow
# The version of PyTorch is not strictly constrained, you can decide according to your own device.
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -U diffusers
pip install transformers accelerate protobuf sentencepiece
```

❗ *2025.04.01*: If you want to try diffusers based video editing using Wan models, you need diffusers' version greater than or equal to `0.33.0.dev0`. These days, it has not been released yet, so here you need to install diffusers through the source code:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```


<h1> 🧪 Examples </h1>

We have prepared some editing examples that you can try using the provided 📰 scripts:

```shell
cd YOUR_WORKSPACE/UniEdit-Flow/UniEdit-Flow_diffusers

# 🖼️ Image Inversion and Reconstruction
sh scripts/inv_rec.sh

# 🎨 Image Editing
sh scripts/edit.sh

# 🎥 Video Editing, please ensure your diffusers' version is acceptable
sh scripts/wan_video_edit
```

The results will be stored in `YOUR_WORKSPACE/UniEdit-Flow/UniEdit-Flow_diffusers/outputs/`.

Moreover, you can also run the following 💻 commands to perform your own experiments:

```shell
cd YOUR_WORKSPACE/UniEdit-Flow/UniEdit-Flow_diffusers

# 🖼️ Image Inversion and Reconstruction 
# Default model path: "stabilityai/stable-diffusion-3-medium-diffusers".
# Null text prompt is used for unconditional inversion and reconstruction, image description is used for conditional inversion and reconstruction.
python demo_inv_rec.py \
    --model_path [your model path] \
    --image_path [your image path] \
    --save_folder [folder you want to save the results] \
    --prompt [image description or null text]

# 🎨 Image Editing
# Default model path: "stabilityai/stable-diffusion-3-medium-diffusers".
# We recommend alpha={0.6, 0.8}, omega=5.0, step={8, 15, 25, 30, 50}.
python demo_edit.py \
    --model_path [your model path] \
    --alpha 0.6 \
    --omega 5.0 \
    --step 15 \
    --image_path [your image path] \
    --save_folder [folder you want to save the results] \
    --src_prompt [image description] \
    --trg_prompt [description of your editing target]

# 🎥 Video Editing, please ensure your diffusers' version is acceptable
# Default model path: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers".
# We recommend alpha={0.6, 0.8}, omega=5.0, step={15, 20, 25, 30, 50}.
python demo_wan_video_edit.py \
    --model_path [your model path] \
    --alpha 0.8 \
    --omega 5.0 \
    --step 25 \
    --video_path [your video path] \
    --save_folder [folder you want to save the results] \
    --src_prompt [video description] \
    --trg_prompt [description of your editing target]
```



<h1> 💡 Easy Utility </h1>

We have integrated Uni-Inv and Uni-Edit into the `scheduler` form of diffusers, so there are some simple uses when your path is `YOUR_WORKSPACE/UniEdit-Flow/UniEdit-Flow_diffusers`:

```python
# Using image editing as an example:
import torch
from PIL import Image 
from diffusers import StableDiffusion3Pipeline
from schedulers.UniInvEulerScheduler import UniInvEulerScheduler
from schedulers.UniEditEulerScheduler import UniEditEulerScheduler
from utils import image2latent, pack_edit_input

# load
model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
invert_scheduler = UniInvEulerScheduler.from_pretrained(model_path, subfolder='scheduler')
edit_scheduler = UniEditEulerScheduler.from_pretrained(model_path, subfolder='scheduler')

# set hyper-parameters
alpha, omega, step = 0.6, 5.0, 15
invert_scheduler.set_hyperparameters(alpha=alpha)
edit_scheduler.set_hyperparameters(alpha=alpha, omega=omega)

# load image 
image_path = ""     # TODO
src_prompt = ""     # TODO
trg_prompt = ""     # TODO
save_path = ""      # TODO
image = Image.open(image_path).convert('RGB')

# invert
pipe.scheduler = invert_scheduler
invert_noise_latent = pipe(
    "",
    num_inference_steps=step,
    latents=image2latent(pipe, image, "cuda", torch.float16),
    output_type='latent',
).images

# edit
pipe.scheduler = edit_scheduler
edit_init_latent, edit_prompt = pack_edit_input(invert_noise_latent, src_prompt, trg_prompt)
edit_image = pipe(
    edit_prompt,
    num_inference_steps=step,
    latents=edit_init_latent.to(torch.float16),
).images[0]
edit_image.save(save_path)
```
