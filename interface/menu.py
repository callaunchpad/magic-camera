from PIL import ImageFont, ImageDraw
from typing import Sequence

FNT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

ITEM_OUTLINE = "#FFFFFF"

ITEM_HEIGHT = 40
ITEM_X_MARGIN = 10
ITEM_Y_MARGIN = 10

class Menu:

    def __init__(self, image_draw: ImageDraw, modes: Sequence[str]):
        assert len(modes) > 0, "must have at least one mode"
        self.image_draw = image_draw
        self.modes = modes
        self.mode_id = 1

    def increment_mode(self):
        self.mode_id = min(self.mode_id + 1, len(self.modes))

    def decrement_mode(self):
        self.mode_id = max(self.mode_id - 1, 0)

    def draw(self):
        for i, mode in enumerate(self.modes):
            self.image_draw.rectangle(
                xy=(ITEM_X_MARGIN, ITEM_Y_MARGIN + i * (ITEM_HEIGHT + ITEM_Y_MARGIN), 240 - ITEM_X_MARGIN, (i+1) * (ITEM_HEIGHT + ITEM_Y_MARGIN)),
                outline=ITEM_OUTLINE,
            )
            self.image_draw.text(
                xy=(ITEM_X_MARGIN + 5, 5 + ITEM_Y_MARGIN + i * (ITEM_HEIGHT + ITEM_Y_MARGIN)),
                text="Hello World",
                font=FNT,
                fill="#ffffff",
            )

