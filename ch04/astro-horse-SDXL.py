import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionXLPipeline
import torch # we need this to set the float

# The expanded path to the model
model = os.path.expanduser("~/Models/stable-diffusion-xl-base-1.0")

# Load the model
pipe = StableDiffusionXLPipeline.from_pretrained(model,
                                  torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
pipe = pipe.to("cpu") # Move the model to the CPU

prompt = "an astronaut riding a horse" # Set the prompt
neg_prompt = "blurry"
image = pipe(prompt, negative_prompt=neg_prompt,
             num_inference_steps=5,
             width=1024, height=1024).images[0]

image.save("astro-horse-SDXL.png")