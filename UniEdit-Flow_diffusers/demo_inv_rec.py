import os 
from pathlib import Path
import argparse

import torch
import numpy as np
from PIL import Image

from diffusers import StableDiffusion3Pipeline, FluxPipeline, DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DDIMScheduler

from schedulers.UniInvEulerScheduler import UniInvEulerScheduler, UniInvDDIMScheduler
from utils import seed_everything, image2latent, latent2image


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Image inversion and reconstruction based on diffusers")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("--prompt", type=str, default="")
    
    parser.add_argument("--step", type=int, default=50, help='1~1000')
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    
    args = parser.parse_args()
    
    # preparation
    os.makedirs(args.save_folder, exist_ok=True)
    seed_everything(args.seed)
    device = 'cuda'
    torch_dtype = torch.float16
    
    # load model and schedulers
    if 'xl' in args.model_path.lower():
        pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        invert_scheduler = UniInvDDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    else:
        if 'flux' in args.model_path.lower():
            pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        elif 'stable-diffusion-3' in args.model_path.lower():
            pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        invert_scheduler = UniInvEulerScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = pipe.to(device)
    
    # load and resize image
    image = Image.open(args.image_path).convert('RGB')
    width, height = image.size
    width = width - width % 16
    height = height - height % 16
    image = image.crop((0, 0, width, height))
    
    # AE encode & decode
    image_latent = image2latent(pipe, image, device, torch_dtype)
    decode_image = latent2image(pipe, image_latent, device, torch_dtype)
    decode_image.save(os.path.join(args.save_folder, 'input.png'))
    
    # invert
    pipe.scheduler = invert_scheduler
    invert_noise_latent = pipe(
        args.prompt,
        num_inference_steps=args.step,
        guidance_scale=1.0,
        latents=image_latent.to(torch_dtype),
        output_type='latent',
        height=height,
        width=width,
    ).images
    invert_noise_image = latent2image(pipe, invert_noise_latent, device, torch_dtype)
    invert_noise_image.save(os.path.join(args.save_folder, 'invert_noise.png'))
    
    # reconstruct
    pipe.scheduler = scheduler
    recon_image = pipe(
        args.prompt,
        num_inference_steps=args.step,
        guidance_scale=1.0,
        latents=invert_noise_latent.to(torch_dtype),
        height=height,
        width=width,
    ).images[0]
    recon_image.save(os.path.join(args.save_folder, 'recon.png'))
    
