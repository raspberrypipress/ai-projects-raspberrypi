import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionXLPipeline
import torch # we need this to set the float

# Load the model
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_name,
                                  torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
pipe = pipe.to("cpu") # Move the model to the CPU
prompt = "an astronaut riding a horse" # Set the prompt
image = pipe(prompt, num_inference_steps=5,
             width=1024, height=1024).images[0]

image.save("astro-horse-SDXL.png")