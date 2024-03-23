import time
import random
from colorsys import hsv_to_rgb
import board
from digitalio import DigitalInOut, Direction
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import st7789

from enum import Enum
from menu import Menu
from utils import *
from viewfinder import Viewfinder

BAUDRATE = 24000000

class Display:

    def __init__(self, modes):
        cs_pin = DigitalInOut(board.CE0)
        dc_pin = DigitalInOut(board.D25)
        reset_pin = DigitalInOut(board.D24)

        spi = board.SPI()
        self.disp = st7789.ST7789(
            spi,
            height=240,
            y_offset=80,
            rotation=180,
            cs=cs_pin,
            dc=dc_pin,
            rst=reset_pin,
            baudrate=BAUDRATE,
        )

        backlight = DigitalInOut(board.D26)
        backlight.switch_to_output()
        backlight.value = True

        self.setup_buttons()

        width = self.disp.width
        height = self.disp.height
        self.image = Image.new("RGB", (width, height))
        self.image_draw = ImageDraw.Draw(self.image)

        # Draw a black filled box to clear the image.
        self.image_draw.rectangle((0, 0, width, height), outline=0, fill=0)

        self.screen = Screen.MENU
        self.menu = Menu(self.image_draw, modes)
        self.viewfinder = Viewfinder(self.image_draw)

    def setup_buttons(self):
        self.button_A = DigitalInOut(board.D5)
        self.button_A.direction = Direction.INPUT

        self.button_B = DigitalInOut(board.D6)
        self.button_B.direction = Direction.INPUT

        self.button_L = DigitalInOut(board.D27)
        self.button_L.direction = Direction.INPUT

        self.button_R = DigitalInOut(board.D23)
        self.button_R.direction = Direction.INPUT

        self.button_U = DigitalInOut(board.D17)
        self.button_U.direction = Direction.INPUT

        self.button_D = DigitalInOut(board.D22)
        self.button_D.direction = Direction.INPUT

        self.button_C = DigitalInOut(board.D4)
        self.button_C.direction = Direction.INPUT
    
    def read_buttons(self):
        if not self.button_A.value:
            print("A pressed")
            if self.screen == Screen.MENU:
                self.screen = Screen.VIEWFINDER
            elif self.screen == Screen.VIEWFINDER:
                pass # take picture

        elif not self.button_B.value:
            print("B pressed")
            if self.screen == Screen.VIEWFINDER:
                self.screen = Screen.MENU

        elif not self.button_U.value:
            print("U pressed")
            if self.screen == Screen.MENU:
                self.menu.decrement_mode()

        elif not self.button_D.value:
            print("D pressed")
            if self.screen == Screen.MENU:
                self.menu.increment_mode()

    def draw(self):
        if self.screen == Screen.MENU:
            self.menu.draw()
        else:
            self.viewfinder.draw()
        self.disp.image(self.image)


if __name__ == "__main__":
    display = Display(modes=["mode one", "mode two"])