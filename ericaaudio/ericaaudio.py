import openai
import os

from openai import OpenAI
from PIL import Image
import audiocraft
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import ffmpeg
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client()

import numpy as np
import requests
import base64
import subprocess
import cv2

def compress_image(input_path, output_path, max_size_mb):
    # Open the image
    image = Image.open(input_path)

    # Calculate the target size in bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    # Compress the image until its size is less than the maximum allowed
    while True:
        # Save the image with a lower quality
        image.save(output_path, quality=85)  # You can adjust the quality as needed
        
        # Check the size of the compressed image
        if os.path.getsize(output_path) < max_size_bytes:
            break
        else:
            # Reduce the quality further if the size is still too large
            image = Image.open(output_path)

# Example usage
input_image_path = "ericaaudio/giraffes.jpeg"
output_image_path = "ericaaudio/compressed_image.jpg"
max_allowed_size_mb = 20


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Getting the base64 string
compress_image(input_image_path, output_image_path, max_allowed_size_mb)
base64_image = encode_image(output_image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Give me a brief phrase describing a type and description of song or audio that would fit this image: only auditory descriptions."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}


response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json())
words = response.json()['choices'][0]['message']['content'].split(', ')
print(words)

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=30)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = words
wav = model.generate(descriptions)  # generates 3 samples.

actual_image = cv2.imread(input_image_path)
output_video = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, 24, (actual_image.shape[1], actual_image.shape[0]))

# Repeat the image for the specified duration
for _ in range(int(30 * 24)):  # Assuming 24 fps
    video_writer.write(actual_image)
video_writer.release()

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write('0', one_wav.cpu(), model.sample_rate, strategy="loudness")

subprocess.run(["ffmpeg", "-y", "-i", "output_video.mp4", "-i", "0.wav", "-c:v", "copy", "chrisp.mp4"])