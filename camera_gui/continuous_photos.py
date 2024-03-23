import time
from picamera import PiCamera
import io

import digitalio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import ili9341
from adafruit_rgb_display import st7789  # pylint: disable=unused-import
from adafruit_rgb_display import hx8357  # pylint: disable=unused-import
from adafruit_rgb_display import st7735  # pylint: disable=unused-import
from adafruit_rgb_display import ssd1351  # pylint: disable=unused-import
from adafruit_rgb_display import ssd1331  # pylint: disable=unused-import

# Configuration for CS and DC pins (these are PiTFT defaults):
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D24)

# Config for display baudrate (default max is 24mhz):
BAUDRATE = 24000000

# Setup SPI bus using hardware SPI:
spi = board.SPI()

# pylint: disable=line-too-long
# Create the display:
# disp = st7789.ST7789(spi, rotation=90,                            # 2.0" ST7789
# disp = st7789.ST7789(spi, height=240, y_offset=80, rotation=180,  # 1.3", 1.54" ST7789
# disp = st7789.ST7789(spi, rotation=90, width=135, height=240, x_offset=53, y_offset=40, # 1.14" ST7789
# disp = st7789.ST7789(spi, rotation=90, width=172, height=320, x_offset=34, # 1.47" ST7789
# disp = st7789.ST7789(spi, rotation=270, width=170, height=320, x_offset=35, # 1.9" ST7789
# disp = hx8357.HX8357(spi, rotation=180,                           # 3.5" HX8357
# disp = st7735.ST7735R(spi, rotation=90,                           # 1.8" ST7735R
# disp = st7735.ST7735R(spi, rotation=270, height=128, x_offset=2, y_offset=3,   # 1.44" ST7735R
# disp = st7735.ST7735R(spi, rotation=90, bgr=True, width=80,       # 0.96" MiniTFT Rev A ST7735R
# disp = st7735.ST7735R(spi, rotation=90, invert=True, width=80,    # 0.96" MiniTFT Rev B ST7735R
# x_offset=26, y_offset=1,
# disp = ssd1351.SSD1351(spi, rotation=180,                         # 1.5" SSD1351
# disp = ssd1351.SSD1351(spi, height=96, y_offset=32, rotation=180, # 1.27" SSD1351
# disp = ssd1331.SSD1331(spi, rotation=180,                         # 0.96" SSD1331
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

backlight = digitalio.DigitalInOut(board.D26)
backlight.switch_to_output()
backlight.value = True
# pylint: enable=line-too-long

width = disp.width
height = disp.height

image = Image.new("RGB", (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)
# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
disp.image(image)

def show_image_on_screen(filename):
    image = Image.open(filename)

    # Scale the image to the smaller screen dimension
    #image_ratio = image.width / image.height
    #screen_ratio = width / height
    #if screen_ratio < image_ratio:
    #    scaled_width = image.width * height // image.height
    #    scaled_height = height
    #else:
    #    scaled_width = width
    #    scaled_height = image.height * width // image.width
    # image = image.resize((scaled_width, scaled_height), Image.BICUBIC)

    # Crop and center the image
    # x = scaled_width // 2 - width // 2
    # y = scaled_height // 2 - height // 2
    # image = image.crop((x, y, x + width, y + height))


    print("displaying to screen rn")
    # Display image.
    disp.image(image)

def show_image(image):
    disp.image(image)

def main():
    with PiCamera() as camera:
        camera.framerate = 15
        
        camera.resolution = (width, height)

        stream = io.BytesIO()
        for _ in camera.capture_continuous(stream, format='jpeg'): 
            print("starting image work")
            # obtain image data
            pil = Image.open(stream)

            # Extract image data and save to file
            # image_data = Image.open(stream)
            show_image(pil)

            stream.seek(0)
            stream.truncate()
            print("done w/ image work")
            # Reset the stream for the next capture
      

def old_main():
    with PiCamera() as camera:
        # Set the desired framerate
        camera.framerate = 5
        camera.resolution = (width, height)

        # Start the continuous capture, specifying output filename pattern
        for filename in camera.capture_continuous('image{counter:03d}.jpg'):
            print(f"Captured image: {filename}")
            show_image_on_screen(filename)

            # Introduce a delay to maintain the 15fps rate
            # time.sleep(1/15)  # Delay for 1/15th of a second

main()

