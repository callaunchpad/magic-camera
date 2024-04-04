import random
import requests
import time
from io import BytesIO
from itertools import count
from typing import Sequence

from PIL import Image, ImageFont

from canvas import Canvas


FNT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

class ImageProcessor:

    def __init__(
        self,
        canvas: Canvas,
        mode_dict: Sequence[str],
        base_url: str,
        verbose: bool = False,
    ):
        assert len(mode_dict) > 0, "must have at least one mode"
        self.canvas = canvas
        self.mode_dict = mode_dict
        self.base_url = base_url
        self.verbose = verbose

        # TODO: read animation paths
        self.animation_paths = ["path1"]
        
        self.success = False
        self.message = ""
        self.result = None

    def animate_loading(self):
        if self.verbose:
            print("\tLOADING")
        animation_path = random.choice(self.animation_paths)
        # TODO: load animation images
        self.canvas.clear_image()
        self.canvas.image_draw.text(
            xy=(0,0),
            text="loading",
            font=FNT,
            fill="#ffffff",
        )
        self.canvas.display_image()
        for t in count(0):
            # TODO: draw frame
            t += 1

    def process_image(self, image: Image, mode_name: str):
        # save image to a file
        # load file and send it through pipeline
        # if pipeline is successful, set self.success to True and put resulting image in self.result
        # oherwise, set self.success to False and put error message in self.message
        if self.verbose:
            print("\tPROCESSING YUH")

        time_str = time.strftime("%Y%m%d-%H%M%S")
        path_before = f"out/{time_str}_before.png"
        path_after = f"out/{time_str}_after.png"

        image.save(path_before)
        with open(path_before, "rb") as f:
            files = {"file": f}
            response = requests.post(self.base_url + self.mode_dict[mode_name], files=files)

        if response.status_code == 200:
            response_json = response.json()
            image_url = response_json.get("url")
            if image_url:
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image = Image.open(BytesIO(image_response.content))
                    image.save(path_after)
                    if self.verbose:
                        print("\tSuccesfully processed image!!")
                else:
                    if self.verbose:
                        print(f"\tFailed to retrieve image: {image_response.status_code}")
            else:
                if self.verbose:
                    print("\tImage URL not found in response.")
        else:
            if self.verbose:
                print(f"\tError with file upload: {response.status_code}, {response.text}")

    def show_result(self):
        self.canvas.clear_image()
        if self.success:
            self.canvas.image_draw.text(
                xy=(0,0),
                text="success",
                font=FNT,
                fill="#ffffff",
            )
        else:
            self.canvas.image_draw.text(
                xy=(0,0),
                text="error",
                font=FNT,
                fill="#ffffff",
            )
        self.canvas.display_image()

    
