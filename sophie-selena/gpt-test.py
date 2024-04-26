from PIL import Image
from openai import OpenAI
import os
import base64
from io import BytesIO

def preprocess(image):
  img = Image.open(image)
  return img

os.getenv("OPENAI_API_KEY") # TODO
client = OpenAI()

i = 0
def gpt_prompt_boxes(description, cropped_img):
  buffered = BytesIO()
  cropped_img.save(buffered, format="JPEG") # TODO
  cropped_img.save(f"out_{i}.jpg")
  i += 1
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
  
  return client.chat.completions.create(
      model="gpt-4-1106-vision-preview",
      messages=messages,
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

def gpt_prompt_caption(descriptions):
  string = ", ".join(descriptions)
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": f"Here are several descriptions of objects: {string}.\nCombine the descriptions to make a prompt for an image-generating stable diffusion model."
        }
      ]
    }
  ]
  return client.chat.completions.create(
      model="gpt-4",
      messages=messages,
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

def create_prompt(image, boxes, phrases):
  new_phrases = []
  for (box, description) in zip(boxes, phrases):
    cropped = image.crop(box)
    new_phrases.append(gpt_prompt_boxes(description, cropped)) # TODO

  prompt = gpt_prompt_caption(new_phrases)

  return prompt, new_phrases

processed_image = Image.open("download.png")
boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
phrases = ["sign", "mango"]
processed_image.crop(boxes[0]).save(f"out_{i}.jpg")
prompt, gpt_phrases = create_prompt(processed_image, boxes, phrases) # TODO
