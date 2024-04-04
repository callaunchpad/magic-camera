import random
import time
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
            print("LOADING")
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

    def process_image(self, image: Image, mode_id: int):
        # save image to a file
        # load file and send it through pipeline
        # if pipeline is successful, set self.success to True and put resulting image in self.result
        # oherwise, set self.success to False and put error message in self.message
        if self.verbose:
            print("PROCESSING YUH")
        time.sleep(5)

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

    
