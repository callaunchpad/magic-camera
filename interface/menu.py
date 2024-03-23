from colorsys import hsv_to_rgb
from PIL import ImageFont, ImageDraw
from typing import Sequence
import random

FNT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

ITEM_OUTLINE = "#FFFFFF"

TOP_PADDING = 15
LEFT_PADDING = 10

ITEM_HEIGHT = 40
ITEM_X_MARGIN = 15
ITEM_Y_MARGIN = 5

POINTER_HEIGHT = 20
POINTER_WIDTH = 15
POINTER_Y_MARGIN = 5

COLORS = [
    "#0000ff",
    "#00ffff",
    "#00ff00",
    "#ff00ff",
    "#ffff00",
    "#ff0000",
]

class Menu:

    def __init__(self, image_draw: ImageDraw, modes: Sequence[str]):
        assert len(modes) > 0, "must have at least one mode"
        self.image_draw = image_draw
        self.modes = modes
        self.selected = 0

        self.randomize_color()

    def randomize_color(self):
        self.color = random.choice(COLORS)

    def increment_mode(self):
        self.selected = min(self.selected + 1, len(self.modes) - 1)
        self.randomize_color()

    def decrement_mode(self):
        self.selected = max(self.selected - 1, 0)
        self.randomize_color()

    def draw(self):
        for i, mode in enumerate(self.modes):
            self.image_draw.text(
                xy=(LEFT_PADDING + POINTER_WIDTH + ITEM_X_MARGIN, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN)),
                text=mode,
                font=FNT,
                fill="#ffffff",
            )

            if self.selected == i:
                self.image_draw.polygon(
                    xy=((LEFT_PADDING, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN) + POINTER_Y_MARGIN),
                        (LEFT_PADDING, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN) + POINTER_Y_MARGIN + POINTER_HEIGHT),
                        (LEFT_PADDING + POINTER_WIDTH, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN) + POINTER_Y_MARGIN + POINTER_HEIGHT/2)),
                    outline="#ffffff",
                    fill=self.color,
                )
