import openai
import os

from openai import OpenAI
from PIL import Image
import audiocraft
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import ffmpeg
from cog import BasePredictor, Input, Path, BaseModel

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client()


import numpy as np
import requests
import base64
import subprocess
import cv2

class Output(BaseModel):
    output_vid: Path
    output_prompt: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass
        

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        )
    ) -> Output:
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

            return Path(output_path)
        
    
        max_allowed_size_mb = 20
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        input_image_path = str(image)
        output_image_path = "imageout.jpeg"

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
        words = response.json()['choices'][0]['message']['content'].split(', ')
        output_string = " ".join(words)

        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=20)  # generate 8 seconds.
        wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
        descriptions = words
        wav = model.generate(descriptions)  # generates 3 samples.
    

        actual_image = cv2.imread(input_image_path)
        output_video = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, 5, (actual_image.shape[1], actual_image.shape[0]))

        # Repeat the image for the specified duration
        for _ in range(int(20 * 5)):  # Assuming 24 fps
            video_writer.write(actual_image)
        video_writer.release()

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write('0', one_wav.cpu(), model.sample_rate, strategy="loudness")
        print("audio write finished")
        #subprocess.run("ffmpeg -i output_video.mp4 -i 0.wav -c:v  copy output.mp4")
        #subprocess.run(["sudo", "ffmpeg", "-i", "output_video.mp4", "-i", "0.wav", "-c:v", "copy", "output2.mp4"])
        #video_stream = ffmpeg.input('output_video.mp4')
        #audio_stream = ffmpeg.input('0.wav')
        #output = ffmpeg.output(video_stream, audio_stream, 'output2.mp4', vcodec='copy', acodec='copy').run()
        subprocess.run(["ffmpeg", "-y", "-i", "output_video.mp4", "-i", "0.wav", "-c:v", "copy", "output10.mp4"])
        #print(output_string)
        return Output(output_vid=Path("output10.mp4"), output_prompt=output_string)
        #ffmpeg.concat(input_video, input_audio, v=1, a=1).output('finished_video.mp4').run()
        #ffmpeg.input('output_video.mp4').input('0.wav').filter('concat', n=2, v=1, a=1).output('output_video_audio.mp4', vcodec='copy', acodec='aac', strict='experimental', ar='44100').run(overwrite_output=True)