import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionXLPipeline
import torch # we need this to set the float

# Get the expanded path to the model
model_name = "stable-diffusion-xl-base-1.0"
model = os.path.expanduser(f"~/Models/{model_name}")

# Load the model
pipe = StableDiffusionXLPipeline.from_pretrained(model,
                                  torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
pipe = pipe.to("cpu") # Move the model to the CPU
prompt = "an astronaut riding a horse" # Set the prompt
image = pipe(prompt, num_inference_steps=5,
             width=1024, height=1024).images[0]

image.save("astro-horse_SDXL.png")