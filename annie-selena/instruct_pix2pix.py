# !pip install -q diffusers accelerate

# import subprocess
# import os
# import sys
# import gc
# import PIL
from PIL import Image
# import requests
import random
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from cog import BasePredictor, Input, Path
import torch

def preprocess(image):
  img = Image.open(image)
  return img

class Predictor(BasePredictor):
  def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    model_id = "timbrooks/instruct-pix2pix" # this guy made sora!!!
    self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    self.pipe.to("cuda")
    self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

  # The arguments and types the model takes as input
  def predict(self,
        image: Path = Input("file path"), 
        prompt: str = None,
        num_inference_steps: int = 10,
        image_guidance_scale: float = 1.2
  ) -> Path:
    """Run a single prediction on the model"""
    processed_image = preprocess(image)
    prompt = random.choice(["change to watercolor style",
                            "change to studio ghibli style",
                            "make it dark and intense",
                            "change to still art style"])
    output = self.pipe(prompt, image=processed_image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale).images
    new_path = f"./out/{prompt}.jpg"
    output[0].save(new_path)
    return new_path