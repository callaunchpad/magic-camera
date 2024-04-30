# Prediction interface for Cog ⚙️
# https://cog.run/python
import os

from cog import BasePredictor, Input, Path
import os
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib
import matplotlib.pyplot as plt
from openai import OpenAI
import replicate
from PIL import ImageFilter
import requests
from dotenv import load_dotenv
load_dotenv()

replicate_key = os.getenv("REPLICATE_API_TOKEN")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass
    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""

            #identify key features in the image by classifying objects in the image
        input = {
            "image": "https://replicate.delivery/pbxt/KRULC43USWlEx4ZNkXltJqvYaHpEx2uJ4IyUQPRPwYb8SzPf/view.jpg",
            "prompt": "Write me a story based on the features in this image"
        }

        output = replicate.run(
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            input=input
        )
        print("".join(output))



        #create outputting meshes


        def frame_maker_transparent(input_image):


            # Get the dimensions of the input image
            width, height = input_image.size

            # Create a new transparent image with double the dimensions
            new_image = Image.new("RGBA", (width * 2, height * 2), (0,0,0,0))

            return new_image

        def frame_maker_white(input_image):


            # Get the dimensions of the input image
            width, height = input_image.size

            # Create a new transparent image with double the dimensions
            new_image = Image.new("RGBA", (width * 2, height * 2), color = "white")

            return new_image


        extended_image = frame_maker_transparent(img)

        extended_image = extended_image.convert("RGBA")
        img = img.convert("RGBA")


        x = (extended_image.width - img.width) // 2
        y = (extended_image.height - img.height) // 2
        extended_image.paste(img, (x, y), img)



        mask_image = Image.new("RGBA", img.size, color="black")

        white_image = frame_maker_white(mask_image)

        white_image = white_image.convert("RGBA")
        mask_image = mask_image.convert("RGBA")

        white_image.paste(mask_image, (x,y), mask_image)

        mask_image = white_image


        extended_image.save("extended.png", "PNG")
        mask_image.save('mask.png', "PNG")


        output = replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={
            "prompt": completion_message,
            "image": open("extended.png", "rb"),
            "mask": open("mask.png", "rb")
        }
        )



        print(output[0])
        print(mask_image.size)



        def point_five(image, distortion_factor):



            width, height = image.size
            center_x, center_y = width / 2, height / 2


            distorted_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, 0), Image.BILINEAR)
            for y in range(height):
                for x in range(width):
                    dx = x - center_x
                    dy = y - center_y
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    if distance < center_x:
                        scale = 1 - distance / center_x
                        new_x = int(center_x + dx * scale * distortion_factor)
                        new_y = int(center_y + dy * scale * distortion_factor)
                        if 0 <= new_x < width and 0 <= new_y < height:
                            distorted_image.putpixel((x, y), image.getpixel((new_x, new_y)))

            return distorted_image




        # URL of the image
        image_url = output[0]

        # Directory where you want to save the image
        save_directory = "/home/ubuntu/magic-camera/arjun"



        # Get the image from the URL
        response = requests.get(image_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract the filename from the URL
            filename = os.path.join(save_directory, "image.png")

            # Save the image to the specified directory
            with open(filename, 'wb') as f:
                f.write(response.content)
            print("Image saved successfully to:", filename)
        else:
            print("Error:", response.status_code)


        generated_image = Image.open("image.png")

        print(generated_image.size)


        full_point_five = point_five(generated_image, 2)

        full_point_five.save("result.png", "PNG")
        
        
        return Path(output_path)





