import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionPipeline
import torch # we need this to set the float
import datetime

# The expanded path to the model
model = os.path.expanduser("~/Models/stable-diffusion-v1-5")

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model,
                                    torch_dtype=torch.float32,
                                    low_cpu_mem_usage=True)
pipe = pipe.to("cpu") # Move the model to the CPU

prompt = "an astronaut riding a horse" # Set the prompt
image = pipe(prompt, num_inference_steps=35, 
             width=640, height=640).images[0]

# add a timecode so we don't overwrite each image.
created = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
image.save("output-" + created + ".png")