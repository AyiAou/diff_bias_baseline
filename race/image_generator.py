from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm

model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda:0")

prompt = "a photo of a white person"

for i in tqdm(range(2000), desc="Generating white images"):
    image = pipe(prompt).images[0]
    image.save(f"./autodl-tmp/white/{i + 1}.png")
    
prompt = "a photo of a black person"

for i in tqdm(range(2000), desc="Generating black images"):
    image = pipe(prompt).images[0]
    image.save(f"./autodl-tmp/black/{i + 1}.png")    
    
prompt = "a photo of an indian"

for i in tqdm(range(2000), desc="Generating indian images"):
    image = pipe(prompt).images[0]
    image.save(f"./autodl-tmp/indian/{i + 1}.png")    
    
prompt = "a photo of an asian"

for i in tqdm(range(2000), desc="Generating asian images"):
    image = pipe(prompt).images[0]
    image.save(f"./autodl-tmp/asian/{i + 1}.png")    
    
