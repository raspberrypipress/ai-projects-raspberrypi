import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionPipeline

# Get the expanded path to the model
model_name = "stable-diffusion-v1-5"
model = os.path.expanduser(f"~/Models/{model_name}")

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model,
                                    low_cpu_mem_usage=True)
pipe = pipe.to("cpu") # Move the model to the CPU

prompt = "an astronaut riding a horse" # Set the prompt
neg_prompt = "blurry"
image = pipe(prompt, negative_prompt=neg_prompt,
             num_inference_steps=5, 
             width=640, height=640).images[0]

image.save("astro-horse.png")