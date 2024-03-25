import random
from itertools import count
from typing import Sequence

from PIL import Image, ImageDraw


class ImageProcessor:

    def __init__(self, image_draw: ImageDraw, modes: Sequence[str], width: int, height: int):
        assert len(modes) > 0, "must have at least one mode"
        self.image_draw = image_draw
        self.modes = modes
        self.width = width
        self.height = height

        # TODO: read animation paths
        self.animation_paths = []
        
        self.success = False
        self.message = ""
        self.result = None

    def animate_loading(self):
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
        pass

    def show_result(self):
        if self.success:
            # draw resulting image
            pass
        else:
            # display error message
            pass

    