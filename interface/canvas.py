from PIL import Image, ImageDraw


class Canvas:

    def __init__(self, disp):
        self.disp = disp
        self.width = self.disp.width
        self.height = self.disp.height
        self.image = Image.new("RGB", (self.width, self.height))
        self.image_draw = ImageDraw.Draw(self.image)

        self.clear_image()

    def display_image(self, image=None):
        if image:
            self.disp.image(image)
        else:
            self.disp.image(self.image)

    def clear_image(self):
        self.image_draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

    def get_image_draw(self):
        return self.image_draw