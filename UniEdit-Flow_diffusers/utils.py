import os

import random 
import torch
import numpy as np
from PIL import Image

from memory_management import load_models_to_gpu, unload_all_models


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

@torch.no_grad()
def image2latent(pipe, image, device, dtype):
    '''image: PIL.Image'''
    image = np.array(image)
    image = torch.from_numpy(image).to(dtype) / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    # input image density range [-1, 1]
    latents = pipe.vae.encode(image)['latent_dist'].mean
    if pipe.vae.config.shift_factor is not None:
        latents = latents * pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    else:
        latents = latents * pipe.vae.config.scaling_factor
    if hasattr(pipe, '_pack_latents'):
        latents = pipe._pack_latents(
            latents,
            1, 
            pipe.transformer.config.in_channels // 4,
            (int(image.shape[-2]) // pipe.vae_scale_factor),
            (int(image.shape[-1]) // pipe.vae_scale_factor),
        )
    return latents


@torch.no_grad()
def latent2image(pipe, latents, device, dtype, custom_shape=None):
    '''return: PIL.Image'''
    if hasattr(pipe, '_unpack_latents'):
        # default square
        if custom_shape is None:
          latents = pipe._unpack_latents(
              latents, 
              int((latents.shape[1] ** 0.5) * pipe.vae_scale_factor) * 2,
              int((latents.shape[1] ** 0.5) * pipe.vae_scale_factor) * 2,
              pipe.vae_scale_factor,
          )
        else:
          latents = pipe._unpack_latents(
              latents, 
              custom_shape[0], custom_shape[1],
              pipe.vae_scale_factor,
          )
        
    latents = latents.to(device).to(dtype)
    if pipe.vae.config.shift_factor is not None:
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    else:
        latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type='pil')[0]
    return image


def pack_edit_input(init_latent, src_prompt, trg_prompt):
    return torch.cat([init_latent, init_latent], dim=0), [src_prompt, trg_prompt]


@ torch.no_grad()
def encode_video(pipe, vae, video):
    unload_all_models(pipe)
    load_models_to_gpu(vae)
    with torch.no_grad():
        video = pipe.video_processor.preprocess_video(video).to('cuda', pipe.vae.dtype)
        latents = pipe.vae.encode(video)['latent_dist'].mean
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(video.device, video.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
            video.device, video.dtype
        )
        latents = (latents - latents_mean) * latents_std
    unload_all_models(vae)
    load_models_to_gpu(pipe)
    return latents
    

@ torch.no_grad()
def decode_video(pipe, vae, latents):
    unload_all_models(pipe)
    load_models_to_gpu(vae)
    with torch.no_grad():
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type='np')
    unload_all_models(vae)
    load_models_to_gpu(pipe)
    return video[0]

