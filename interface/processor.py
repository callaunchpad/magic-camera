import random
import requests
import time
from io import BytesIO
from itertools import count
from typing import Sequence
import os
import shutil

from PIL import Image, ImageFont

from canvas import Canvas
from process_when_wifi import check_wifi


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

    def animate_loading(self):
        animation_path = random.choice(self.animation_paths)
        # TODO: load animation images
        self.canvas.clear_image()
        randText = texts[random.randrange(len(texts))]
        self.canvas.image_draw.text(
            xy=(50,100),
            text="loading...",
            font=FNT,
            fill="#ffffff",
        )
        self.canvas.display_image()
        for t in count(0):
            # TODO: draw frame
            t += 1

    def set_image_target(self, image: Image, mode_name: str):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        mode_endpoint_name = self.mode_dict[mode_name]
        self.path_before = f"out/{time_str}_{mode_endpoint_name}_before.png"
        self.path_after = f"out/{time_str}_{mode_endpoint_name}_after.png"
        image.save(self.path_before)

    def process_image(self, mode_name: str):
        print(f"\tprocessing image with mode {mode_name}")
        start_time = time.time()
        with open(self.path_before, "rb") as f:
            files = {"file": f}
            if check_wifi():
                try:
                    response = requests.post(self.base_url + self.mode_dict[mode_name], files=files)
                    response.raise_for_status()
                    return
                except requests.exceptions.RequestException as e:
                    print(f"\tfailed to upload image: {e}")
            # save path in a processing directory
            os.makedirs("to_process", exist_ok=True)
            # move `self.path_before` to `to_process`
            shutil.move(self.path_before, "to_process")
            return

        """
        if response.status_code == 200:
            response_json = response.json()
            image_url = response_json.get("url")
            if image_url:
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    result_image = Image.open(BytesIO(image_response.content))
                    result_image.save(self.path_after)
                    if self.verbose:
                        print("\tsuccesfully processed image!!")
                else:
                    if self.verbose:
                        print(f"\tfailed to retrieve image: {image_response.status_code}")
            else:
                if self.verbose:
                    print("\timage URL not found in response")
        else:
            if self.verbose:
                print(f"\terror with file upload: {response.status_code}, {response.text}")
        
        if self.verbose:
            print(f"\tfinished processing after {(time.time()-start_time)/60:.2f} minutes")
        """

    def show_result(self):
        self.canvas.clear_image()
        try:
            result_image = Image.open(self.path_after)
            result_image = result_image.resize((self.canvas.width, self.canvas.height))
            self.canvas.display_image(result_image)
        except Exception as e:
            print(f"\tboooooo an error: {e}")
            self.canvas.image_draw.text(
                xy=(0,0),
                text="error",
                font=FNT,
                fill="#ffffff",
            )
            self.canvas.display_image()

    
