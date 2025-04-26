import os 
from pathlib import Path
import argparse

import torch
import numpy as np
from PIL import Image

from diffusers import StableDiffusion3Pipeline, FluxPipeline, DiffusionPipeline

from schedulers.UniInvEulerScheduler import UniInvEulerScheduler, UniInvDDIMScheduler
from schedulers.UniEditEulerScheduler import UniEditEulerScheduler, UniEditDDIMScheduler
from utils import seed_everything, image2latent, latent2image, pack_edit_input


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Image editing based on diffusers")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("--src_prompt", type=str, default="")
    parser.add_argument("--trg_prompt", type=str, default="")
    
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--omega", type=float, default=5.0)
    
    parser.add_argument("--step", type=int, default=15, help='1~1000')
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
        invert_scheduler = UniInvDDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        edit_scheduler = UniEditDDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    else:
        if 'flux' in args.model_path.lower():
            pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype)
        invert_scheduler = UniInvEulerScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        edit_scheduler = UniEditEulerScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = pipe.to(device)
    
    # set hyper-parameters
    invert_scheduler.set_hyperparameters(alpha=args.alpha)
    edit_scheduler.set_hyperparameters(alpha=args.alpha, omega=args.omega)
    
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
        "",
        num_inference_steps=args.step,
        guidance_scale=1.0,
        latents=image_latent.to(torch_dtype),
        output_type='latent',
        height=height,
        width=width,
    ).images
    
    # obtain input for editing
    edit_init_latent, edit_prompt = pack_edit_input(invert_noise_latent, args.src_prompt, args.trg_prompt)
    
    # edit
    pipe.scheduler = edit_scheduler
    recon_image = pipe(
        edit_prompt,
        num_inference_steps=args.step,
        guidance_scale=1.0,
        latents=edit_init_latent.to(torch_dtype),
        height=height,
        width=width,
    ).images[0]
    
    # save
    invert_noise_image = latent2image(pipe, invert_noise_latent, device, torch_dtype)
    invert_noise_image.save(os.path.join(args.save_folder, 'invert_noise.png'))
    recon_image.save(os.path.join(args.save_folder, 'result_edit.png'))
    
