import io
import board
from enum import Enum

from adafruit_rgb_display import st7789
from digitalio import DigitalInOut, Direction
from picamera import PiCamera
from PIL import Image, ImageDraw, ImageFont

from menu import Menu


BAUDRATE = 24000000

class Screen(Enum):
    MENU = 0
    VIEWFINDER = 1


class Display:

    def __init__(self, modes, verbose=False):
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

        self.width = self.disp.width
        self.height = self.disp.height
        self.image = Image.new("RGB", (self.width, self.height))
        self.image_draw = ImageDraw.Draw(self.image)

        # Draw a black filled box to clear the image.
        self.image_draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        self.screen = Screen.MENU
        self.menu = Menu(self.image_draw, modes, self.width, self.height)

        self.verbose = verbose

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
            if self.verbose: print("A pressed")
            if self.screen == Screen.VIEWFINDER:
                self.screen = Screen.MENU

        elif not self.button_B.value:
            if self.verbose: print("B pressed")
            if self.screen == Screen.MENU:
                self.screen = Screen.VIEWFINDER
            elif self.screen == Screen.VIEWFINDER:
                pass # take picture

        elif not self.button_U.value:
            if self.verbose: print("U pressed")
            if self.screen == Screen.MENU:
                self.menu.decrement_mode()

        elif not self.button_D.value:
            if self.verbose: print("D pressed")
            if self.screen == Screen.MENU:
                self.menu.increment_mode()

    def run(self):
        while True:
            if self.screen == Screen.MENU: self.run_menu()
            elif self.screen == Screen.VIEWFINDER: self.run_viewfinder()

    def run_menu(self):
        while True:
            self.read_buttons()
            if self.screen == Screen.MENU:
                self.menu.draw()
                self.disp.image(self.image)
            else:
                return
            
    def run_viewfinder(self):
        stream = io.BytesIO()
        with PiCamera() as camera:
            camera.framerate = 15
            camera.resolution = (self.width, self.height)
            for _ in camera.capture_continuous(stream, format='jpeg'): 
                self.read_buttons()
                if self.screen == Screen.VIEWFINDER:
                    camera_img = Image.open(stream)
                    self.disp.image(camera_img)
                    stream.seek(0)
                    stream.truncate()
                else:
                    return
