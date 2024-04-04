import io
import requests
import time
from enum import Enum
from multiprocessing import Process

import board
from adafruit_rgb_display import st7789
from digitalio import DigitalInOut, Direction
from picamera import PiCamera
from PIL import Image, ImageDraw

from canvas import Canvas
from image_processor import ImageProcessor
from menu import Menu


BAUDRATE = 24000000
BASE_URL = "http://52.25.237.192:8000/"

class Screen(Enum):
    MENU = 0
    VIEWFINDER = 1
    LOADING = 2
    RESULT = 3


class Display:

    def __init__(self, verbose=False):
        self.verbose = verbose

        cs_pin = DigitalInOut(board.CE0)
        dc_pin = DigitalInOut(board.D25)
        reset_pin = DigitalInOut(board.D24)

        spi = board.SPI()
        disp = st7789.ST7789(
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
        self.canvas = Canvas(disp)
        self.screen = Screen.MENU
        
        mode_dict = self.get_modes()
        mode_names = list(mode_dict.keys())
        self.menu = Menu(self.canvas, mode_names)
        self.processor = ImageProcessor(self.canvas, mode_dict, BASE_URL, verbose=verbose)
        
        self.last_button_press = 0
        self.camera_res = (self.canvas.width, self.canvas.height)

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
        if time.time() - self.last_button_press < 1.0:
            return

        if not self.button_A.value:
            if self.verbose: print("A pressed")
            if self.screen == Screen.VIEWFINDER:
                self.screen = Screen.MENU
            elif self.screen == Screen.RESULT:
                self.screen = Screen.VIEWFINDER

        elif not self.button_B.value:
            if self.verbose: print("B pressed")
            if self.screen == Screen.MENU:
                self.screen = Screen.VIEWFINDER
            elif self.screen == Screen.VIEWFINDER:
                self.screen = Screen.LOADING # take a picture
            elif self.screen == Screen.RESULT:
                self.screen = Screen.MENU

        elif not self.button_U.value:
            if self.verbose: print("U pressed")
            if self.screen == Screen.MENU:
                self.menu.decrement_mode()

        elif not self.button_D.value:
            if self.verbose: print("D pressed")
            if self.screen == Screen.MENU:
                self.menu.increment_mode()

        else:
            return

        self.last_button_press = time.time()

    def get_modes(self):
        response = requests.get(BASE_URL + "endpoints")
        if response.status_code == 200:
            modes = response.json()
        else:
            modes = {"Jimmy-Inator": "jimmyinator"} # backup mode
        if self.verbose:
            print(f"loaded modes: {modes}")
        return modes

    def run(self):
        while True:
            if self.screen == Screen.MENU:
                if self.verbose: print("running menu screen")
                self.run_menu()
            elif self.screen == Screen.VIEWFINDER:
                if self.verbose: print("running viewfinder screen")
                self.run_viewfinder()
            elif self.screen == Screen.LOADING:
                if self.verbose: print("running loading screen")
                self.run_loading()
            elif self.screen == Screen.RESULT:
                if self.verbose: print("running result screen")
                self.run_result()

    def run_menu(self):
        while True:
            self.read_buttons()
            if self.screen == Screen.MENU:
                self.menu.draw()
            else:
                return
            
    def run_viewfinder(self):
        stream = io.BytesIO()
        with PiCamera() as camera:
            camera.framerate = 15
            # camera.resolution = (self.width, self.height)
            camera.resolution = self.camera_res
            for _ in camera.capture_continuous(stream, format='jpeg'): 
                self.read_buttons()
                if self.screen == Screen.VIEWFINDER:
                    self.camera_img = Image.open(stream)
                    self.canvas.display_image(self.camera_img)
                    stream.seek(0)
                    stream.truncate()
                else:
                    return
                
    def run_loading(self):
        # TODO: make sure self.camera_img exists
        p1 = Process(target=self.processor.process_image, args=(self.camera_img, self.menu.selected))
        p1.start()
        
        p2 = Process(target=self.processor.animate_loading)
        p2.start()

        p1.join()
        self.screen = Screen.RESULT

    def run_result(self):
        while True:
            self.read_buttons()
            if self.screen == Screen.RESULT:
                self.processor.show_result()
            else:
                return
