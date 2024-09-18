from diffusers import StableDiffusionPipeline
import torch

model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda:0")

prompt = "a photo of a male"

for i in range(5000):
    image = pipe(prompt).images[0]
    image.save(f"./autodl-tmp/male/{i + 1}.png")
    
prompt = "a photo of a female"

for i in range(5000):
    image = pipe(prompt).images[0]
    image.save(f"./autodl-tmp/female/{i + 1}.png")
