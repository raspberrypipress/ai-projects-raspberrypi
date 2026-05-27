import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionPipeline
from PIL import Image

# The expanded path to the model
model = os.path.expanduser("~/Models/stable-diffusion-v1-5")

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model,
                                    low_cpu_mem_usage=True)
pipe = pipe.to("cpu") # Move the model to the CPU

prompt = "an astronaut riding a horse" # Set the prompt
image = pipe(prompt, num_inference_steps=35, 
             width=640, height=640).images[0]

import pdb; pdb.set_trace() # Set a breakpoint here to inspect the image variable

image.save("astro-horse.png")