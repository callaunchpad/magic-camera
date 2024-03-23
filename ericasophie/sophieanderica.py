import os

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt


#path = os.path.join(DATA_PATH, "giraffes.jpeg")
image = cv2.imread("giraffes.jpeg")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale_image

#path2 = os.path.join(DATA_PATH, "dinosaurs.jpg")
target_image = cv2.imread("dinosaurs.jpg")

replacement_background = cv2.resize(target_image, (grayscale_image.shape[1], grayscale_image.shape[0]))

threshold_value = 105
binary_mask = (grayscale_image > threshold_value).astype(np.uint8)
inverse = 1-binary_mask



plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='binary')
plt.title('Binary Mask')

plt.show()

#background_color = [0, 255, 255]

# Create a 3-channel mask by stacking the binary mask three times
color_mask = np.stack([inverse] * 3, axis=-1)

result_image = np.where(color_mask == 1, image, replacement_background)
#print(result_image.shape)

converted_image = cv2.convertScaleAbs(result_image)
Image.fromarray(converted_image).save("resultimage.jpeg")

"""plt.imshow(cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Binary Mask')
plt.show()"""

