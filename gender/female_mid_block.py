import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os

cnt = 0
mid_block_out = []
device = "cuda"
def forward_hook(module, input, output):
    mid_block_out.append(output.detach().cpu())

@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    latents = start_latents.clone()

    intermediate_latents = []

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))
        next_t = t
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")

    if size is not None:
        img = img.resize(size)

    return img

model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

mid_block = pipe.unet.mid_block
hook = mid_block.register_forward_hook(forward_hook)

input_image_prompt = "a photo of a female"

folder_path = './autodl-tmp/female'

for filename in os.listdir(folder_path):
    input_image = load_image(os.path.join(folder_path, filename), size=(512, 512))
    base_filename = os.path.splitext(filename)[0]
    output_folder = f'./autodl-tmp/female_outputs/{base_filename}'
    os.makedirs(output_folder, exist_ok=True)
    # input_image.show()
    with torch.no_grad():
        latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device).half() * 2 - 1)
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=50)
    for step, output in enumerate(mid_block_out):
        filename_save = os.path.join(output_folder, f"step_{step + 1}.pt")
        torch.save(output, filename_save)
    mid_block_out.clear()
