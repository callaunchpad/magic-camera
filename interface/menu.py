from PIL import ImageFont

FNT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)

class Menu:

    def __init__(self, image_draw, modes):
        assert len(modes) > 0, "must have at least one mode"
        self.image_draw = image_draw
        self.modes = modes
        self.mode_id = 1

    def increment_mode(self):
        self.mode_id = min(self.mode_id + 1, len(self.modes))

    def decrement_mode(self):
        self.mode_id = max(self.mode_id - 1, 0)

    def draw(self):
        pass

