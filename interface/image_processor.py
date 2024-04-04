import random
import time
from itertools import count
from typing import Sequence

from PIL import Image

from canvas import Canvas


class ImageProcessor:

    def __init__(self, canvas: Canvas, modes: Sequence[str]):
        assert len(modes) > 0, "must have at least one mode"
        self.canvas = canvas
        self.modes = modes

        # TODO: read animation paths
        self.animation_paths = ["path1"]
        
        self.success = False
        self.message = ""
        self.result = None

    def animate_loading(self):
        print("LOADING")
        self.canvas.clear_image()
        animation_path = random.choice(self.animation_paths)
        # TODO: load animation images
        for t in count(0):
            # TODO: draw frame
            t += 1

    def process_image(self, image: Image, mode_id: int):
        # save image to a file
        # load file and send it through pipeline
        # if pipeline is successful, set self.success to True and put resulting image in self.result
        # oherwise, set self.success to False and put error message in self.message
        print("PROCESSING YUH")
        time.sleep(5)

    def show_result(self):
        self.canvas.clear_image()
        if self.success:
            # draw resulting image
            pass
        else:
            # display error message
            pass

    
