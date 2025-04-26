import os 
import numpy as np 
from pathlib import Path 
import argparse

import torch
from diffusers.utils import export_to_video, load_video
from diffusers import AutoencoderKLWan, WanPipeline

from schedulers.UniInvEulerScheduler import UniInvEulerScheduler
from schedulers.UniEditEulerScheduler import UniEditEulerScheduler
from utils import seed_everything, pack_edit_input, encode_video, decode_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Video editing based on diffusers")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("--src_prompt", type=str, default="")
    parser.add_argument("--trg_prompt", type=str, default="")
    
    parser.add_argument("--flow_shift", type=float, default=3.0)
    
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--omega", type=float, default=5.0)
    
    parser.add_argument("--step", type=int, default=15, help='1~1000')
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    
    args = parser.parse_args()
    
    # preparation
    os.makedirs(args.save_folder, exist_ok=True)
    seed_everything(args.seed)
    device = 'cuda'
    
    # load VAE and pipeline separately to avoid OOM
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to('cuda')
    
    # load scheduler
    invert_scheduler = UniInvEulerScheduler(num_train_timesteps=1000, shift=args.flow_shift)
    edit_scheduler = UniEditEulerScheduler(num_train_timesteps=1000, shift=args.flow_shift)
    
    # set hyper-parameters
    invert_scheduler.set_hyperparameters(alpha=args.alpha)
    edit_scheduler.set_hyperparameters(alpha=args.alpha, omega=args.omega)
    
    # load video
    video = load_video(args.video_path)
    height = video[0].size[1]
    width = video[0].size[0]
    num_frames = len(video)
    
    # AE
    video_latents = encode_video(pipe, vae, video)
    ae_video = decode_video(pipe, vae, video_latents)
    export_to_video(
        ae_video, 
        os.path.join(args.save_folder, 'ae_video.mp4'), 
        fps=args.fps
    )
    
    # invert
    pipe.scheduler = invert_scheduler
    invert_noise_latents = pipe(
        prompt="",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=args.step,
        guidance_scale=1.0,
        output_type='latent',
        latents=video_latents,
    ).frames
    invert_noise_video = decode_video(pipe, vae, invert_noise_latents)
    export_to_video(
        invert_noise_video, 
        os.path.join(args.save_folder, 'invert_noise_video.mp4'), 
        fps=args.fps
    )
    
    # edit
    edit_init_latent, edit_prompt = pack_edit_input(invert_noise_latents, args.src_prompt, args.trg_prompt)
    pipe.scheduler = edit_scheduler
    edit_latents = pipe(
        prompt=edit_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=args.step,
        guidance_scale=1.0,
        output_type='latent',
        latents=edit_init_latent,
        ).frames
    edit_video = decode_video(pipe, vae, edit_latents[: 1, ...])
    export_to_video(
        edit_video, 
        os.path.join(args.save_folder, 'result_edit_video.mp4'), 
        fps=args.fps
    )

