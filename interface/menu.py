import random
from PIL import ImageFont
from typing import Sequence

from canvas import Canvas


FNT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

ITEM_OUTLINE = "#FFFFFF"

TOP_PADDING = 15
LEFT_PADDING = 10

ITEM_HEIGHT = 40
ITEM_X_MARGIN = 15
ITEM_Y_MARGIN = 5
NUM_ITEMS = 5

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

    def __init__(self, canvas: Canvas, mode_names: Sequence[str]):
        assert len(mode_names) > 0, "must have at least one mode"
        self.canvas = canvas
        self.mode_names = mode_names

        self.selected = 0
        self.scroll_index = 0
        self.randomize_color()

    def randomize_color(self):
        self.color = random.choice(COLORS)

    def increment_mode(self):
        self.randomize_color()
        self.selected = min(self.selected + 1, len(self.mode_names) - 1)
        if self.selected >= self.scroll_index + NUM_ITEMS:
            self.scroll_index += 1
        
    def decrement_mode(self):
        self.randomize_color()
        self.selected = max(self.selected - 1, 0)
        if self.selected < self.scroll_index:
            self.scroll_index -= 1

    def get_current_mode(self):
        return self.mode_names[self.selected]

    def draw(self):
        self.canvas.clear_image()
        for i, mode in enumerate(self.mode_names[self.scroll_index : self.scroll_index + NUM_ITEMS]):
            self.canvas.image_draw.text(
                xy=(LEFT_PADDING + POINTER_WIDTH + ITEM_X_MARGIN, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN)),
                text=mode,
                font=FNT,
                fill="#ffffff",
            )

            if self.selected == i:
                self.canvas.image_draw.polygon(
                    xy=((LEFT_PADDING, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN) + POINTER_Y_MARGIN),
                        (LEFT_PADDING, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN) + POINTER_Y_MARGIN + POINTER_HEIGHT),
                        (LEFT_PADDING + POINTER_WIDTH, TOP_PADDING + i * (ITEM_HEIGHT + ITEM_Y_MARGIN) + POINTER_Y_MARGIN + POINTER_HEIGHT/2)),
                    outline="#ffffff",
                    fill=self.color,
                )
        self.canvas.display_image()
