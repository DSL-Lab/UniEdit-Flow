# coding=utf-8
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import torch
import numpy as np

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DDIMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class UniEditEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass
class UniEditDDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class UniEditEulerScheduler(FlowMatchEulerDiscreteScheduler):
    omega=5
    alpha=0.6

    def set_hyperparameters(self, omega=5, alpha=1):
        self.omega = omega
        self.alpha = alpha
    
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)
        self.num_inference_steps = num_inference_steps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        
        self.timesteps = timesteps.to(device=device)
        self.sigmas = sigmas
        
        if self.alpha < 1:
            sample_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - sample_steps
            self.timesteps = self.timesteps[skip_steps: ]
            self.sigmas = self.sigmas[skip_steps: ]
            
        self._step_index = 0
        self._begin_index = 0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[UniEditEulerSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
            
        # batch level chunk
        v_src, v_trg = model_output.chunk(2, dim=0)
        guidance = v_trg - v_src
        batch_size = guidance.shape[0]
        
        # mask, [B, C, H, W] for SD3, [B, N, C] for FLUX
        if len(guidance.shape) == 4:
            mask = guidance.mean(dim=1, keepdim=True)
            mask_min = mask.reshape(batch_size, -1).min(dim=1).values.reshape(batch_size, 1, 1, 1)
            mask_max = mask.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1, 1)
        elif len(guidance.shape) == 5:
            mask = guidance.mean(dim=1, keepdim=True)
            mask_min = mask.reshape(batch_size, -1).min(dim=1).values.reshape(batch_size, 1, 1, 1, 1)
            mask_max = mask.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1, 1, 1)
        elif len(guidance.shape) == 3:
            mask = guidance.mean(dim=2, keepdim=True)
            mask_min = mask.reshape(batch_size, -1).min(dim=1).values.reshape(batch_size, 1, 1)
            mask_max = mask.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1)
        mask = (mask - mask_min) / (mask_max - mask_min + 1e-7)
        
        # correction
        stride_corr = self.omega * (sigma_next - sigma) * (1 + mask) * guidance
        
        # velocity fusion
        velocity_fusion = mask * v_trg + (1 - mask) * v_src

        stride_corr = torch.cat([stride_corr, stride_corr], dim=0)
        velocity_fusion = torch.cat([velocity_fusion, velocity_fusion], dim=0)
        prev_sample = sample + stride_corr + (sigma_next - sigma) * velocity_fusion

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return UniEditEulerSchedulerOutput(prev_sample=prev_sample)


class UniEditDDIMScheduler(DDIMScheduler):
    omega=5
    alpha=0.6

    def set_hyperparameters(self, omega=5, alpha=1):
        self.omega = omega
        self.alpha = alpha
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)
        
        if self.alpha < 1:
            sample_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - sample_steps
            self.timesteps = self.timesteps[skip_steps: ]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UniEditDDIMSchedulerOutput, Tuple]:
        
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        
        # UniEdit
        v_src, v_trg = model_output.chunk(2, dim=0)
        guidance = v_trg - v_src
        batch_size = guidance.shape[0]

        if len(guidance.shape) == 4:
            mask = guidance.mean(dim=1, keepdim=True)
            mask_min = mask.reshape(batch_size, -1).min(dim=1).values.reshape(batch_size, 1, 1, 1)
            mask_max = mask.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1, 1)
        elif len(guidance.shape) == 5:
            mask = guidance.mean(dim=1, keepdim=True)
            mask_min = mask.reshape(batch_size, -1).min(dim=1).values.reshape(batch_size, 1, 1, 1, 1)
            mask_max = mask.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1, 1, 1)
        elif len(guidance.shape) == 3:
            mask = guidance.mean(dim=2, keepdim=True)
            mask_min = mask.reshape(batch_size, -1).min(dim=1).values.reshape(batch_size, 1, 1)
            mask_max = mask.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1)
        mask = (mask - mask_min) / (mask_max - mask_min + 1e-7)
        
        # correction
        velocity_corr = self.omega * (1 + mask) * guidance
        
        # velocity fusion
        velocity_fusion = mask * v_trg + (1 - mask) * v_src
        
        # combine at velocity level
        model_output = velocity_corr + velocity_fusion
        model_output = torch.cat([model_output, model_output], dim=0)
        
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return UniEditDDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
