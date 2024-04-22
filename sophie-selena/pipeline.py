import torch
from diffusers import StableDiffusionGLIGENPipeline
from cog import BasePredictor, Input, Path
from PIL import Image
from openai import OpenAI
import os
import base64
from io import BytesIO

def preprocess(image):
  img = Image.open(image)
  return img

class Predictor(BasePredictor):
  def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    model_id = "masterful/gligen-1-4-generation-text-box"
    self.pipe = StableDiffusionGLIGENPipeline.from_pretrained(
      model_id, variant="fp16", torch_dtype=torch.float16
    )
    self.pipe.to("cuda")
    os.getenv("OPENAI_API_KEY") # TODO
    self.client = OpenAI()

  def gpt_prompt_boxes(self, description, cropped_img):
    buffered = BytesIO()
    cropped_img.save(buffered, format="JPEG") # TODO
    encoded_image = base64.b64encode(buffered.getvalue())
    messages = [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Here is an image with the description {description}. Describe the image to make a prompt for a stable diffusion model."
            },
            {
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{encoded_image}"
              }
            }
          ]
        }
      ]
    
    return self.client.chat.completions.create(
        model="gpt-4-1106-vision-preview",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

  def gpt_prompt_caption(self, descriptions):
    messages = [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"Here are several descriptions of objects: {', '.join(descriptions)}.\nCombine the descriptions to make a prompt for an image-generating stable diffusion model."
          }
        ]
      }
    ]
    return self.client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

  def create_prompt(self, image, boxes, phrases):
    new_phrases = []
    for (box, description) in zip(boxes, phrases):
      cropped = image.crop(box)
      new_phrases.append(self.gpt_prompt_boxes(description, cropped)) # TODO

    prompt = self.gpt_prompt_caption(new_phrases)

    return prompt, new_phrases

  # The arguments and types the model takes as input
  def predict(self,
        image: Path = Input("file path"), 
        num_inference_steps: int = 10,
  ) -> Path:
    """Run a single prediction on the model"""
    processed_image = preprocess(image)
    # TODO take in processed_image
    boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
    phrases = ["sign", "mango"]
    prompt, gpt_phrases = self.create_prompt(processed_image, boxes, phrases) # TODO
    output = self.pipe(
      prompt,
      gligen_phrases=gpt_phrases,
      gligen_boxes=boxes,
      gligen_scheduled_sampling_beta=1,
      output_type="pil",
      num_inference_steps=num_inference_steps,
    ).images
    new_path = "./out.jpg" 
    output[0].save(new_path)
    return Path(new_path)
