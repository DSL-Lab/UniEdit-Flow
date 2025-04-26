import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image


import torch
import gradio as gr

from flux.sampling import edit_uniedit, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5)

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


class FluxEditor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload
        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'result'
        self.add_sampling_metadata = True

        if self.name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {self.name}, chose from {available}")

        # init all components
        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.name, device="cpu" if self.offload else self.device)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)
        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
    
    @torch.inference_mode()
    def edit(
        self, init_image, source_prompt, target_prompt, 
        alpha, omega,
        num_steps, guidance,
    ):
        torch.cuda.empty_cache()
        seed = None
        zero_init = False
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)
        
        shape = init_image.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:new_h, :new_w, :]
        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, self.device, self.ae)
        print(init_image.shape)

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        opts.seed = None
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        info = {}
        info['inject_step'] = 0
        info['editing_strategy']= ""
        info['start_layer_index'] = 0
        info['end_layer_index'] = 0
        info['reuse_v']= False
        qkv_ratio = '1.0,1.0,1.0'
        info['qkv_ratio'] = list(map(float, qkv_ratio.split(',')))
        
        # UniEdit-Flow
        info['alpha'] = alpha
        info['omega'] = omega
        info['zero_init'] = zero_init

        inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
        inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
        # src cond
        src_tmp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
        inp_target['src_txt'] = src_tmp['txt']
        inp_target['src_txt_ids'] = src_tmp['txt_ids']
        inp_target['src_vec'] = src_tmp['vec']
        
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # inversion initial noise
        z, info = edit_uniedit(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))

        # denoise initial noise
        x, _ = edit_uniedit(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.width, opts.height)

        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        if self.add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = source_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)

        print("End Edit")
        return img



def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor(args)
    is_schnell = model_name == "flux-schnell"
    
    # Pre-defined examples
    examples = [
        ["gradio_examples/dog.jpg", "Photograph of a dog on the grass", "Photograph of a cat on the grass", 0.7, 5, 12, 1],
        ["gradio_examples/gold.jpg", "a flower in the style of 3d melting gold render", "a cat in the style of 3d melting gold render", 0.8, 5, 15, 1],
        ["gradio_examples/boy.jpg", "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above.", "A young boy is sitting on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above.", 0.75, 5, 12, 1],
        ["gradio_examples/cartoon.jpg", "a cartoon style young boy raising his left hand", "a cartoon style Albert Einstein raising his left hand", 0.6, 5, 8, 1.5],
        ["gradio_examples/nobel.jpg", "A minimalistic line-drawing portrait of Geoffrey Hinton with black lines and light brown shadow", "A minimalistic line-drawing portrait of Donald Trump with black lines and light brown shadow", 0.8, 5, 12, 1],
        ["gradio_examples/art.jpg", "", "a vivid depiction of the Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art.", 0.8, 5, 10, 1],
    ]

    with gr.Blocks() as demo:
        gr.Markdown(f"# UniEdit-Flow Demo (FLUX for image editing)")
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value="Describe the content of the uploaded image.")
                target_prompt = gr.Textbox(label="Target Prompt", value="Describe the desired content of the edited image.")
                init_image = gr.Image(label="Input Image", visible=True)
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    alpha = gr.Slider(0.0, 1.0, 0.6, step=0.05, label=f"Delay rate (α)")
                    omega = gr.Slider(2.0, 10.0, 5, step=0.5, label=f"Guidance strength (ω)")
                    num_steps = gr.Slider(1, 50, 15, step=1, label="Number of steps")
                    guidance = gr.Slider(1.0, 2.0, 1, step=0.5, label="CFG Guidance (Optional)", interactive=not is_schnell)
                output_image = gr.Image(label="Generated Image")

        generate_btn.click(
            fn=editor.edit,
            inputs=[init_image, source_prompt, target_prompt, alpha, omega, num_steps, guidance],
            outputs=[output_image]
        )
        
        # Add examples
        gr.Examples(
            examples=examples,
            inputs=[
                init_image, 
                source_prompt, 
                target_prompt, 
                alpha, 
                omega, 
                num_steps, 
                guidance
            ],
            outputs=[output_image],
            fn=editor.edit,
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=41035)
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
