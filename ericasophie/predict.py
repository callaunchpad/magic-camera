# Prediction interface for Cog ⚙️
# https://cog.run/python
import os

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from cog import BasePredictor, Input, Path



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass
        

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        inputImage = cv2.imread(str(image))
        b, g, r = cv2.split(inputImage)
        inputImage = cv2.merge([r, g, b])
        grayscale_image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)


        #path2 = os.path.join(DATA_PATH, "dinosaurs.jpg")
        target_image = cv2.imread("dinosaurs.jpg")
        b, g, r = cv2.split(target_image)
        target_image = cv2.merge([r, g, b])

        replacement_background = cv2.resize(target_image, (grayscale_image.shape[1], grayscale_image.shape[0]))

        threshold_value = 105
        binary_mask = (grayscale_image > threshold_value).astype(np.uint8)
        inverse = 1-binary_mask
        # Create a 3-channel mask by stacking the binary mask three times
        color_mask = np.stack([inverse] * 3, axis=-1)

        result_image = np.where(color_mask == 1, inputImage, replacement_background).astype(float)
        print(result_image.dtype)
        print(result_image.shape)
        converted_image = cv2.convertScaleAbs(result_image)
        output_path = "resultimage.jpeg"
        Image.fromarray(converted_image).save("resultimage.jpeg")
        return Path(output_path)