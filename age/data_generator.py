from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm

model_id = "./stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = "a photo of a young person"

# for i in tqdm(range(2000), desc="Generating young images"):
#     image = pipe(prompt).images[0]
#     image.save(f"./data/young/{i + 1}.png")
    
prompt = "a photo of an adult person"

for i in tqdm(range(2000), desc="Generating adult images"):
    image = pipe(prompt).images[0]
    image.save(f"./data/adult/{i + 1}.png")    
    
# prompt = "a photo of an old person"

# for i in tqdm(range(2000), desc="Generating old images"):
#     image = pipe(prompt).images[0]
#     image.save(f"./data/old/{i + 1}.png")    
    

