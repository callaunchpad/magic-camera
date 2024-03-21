import os
from picamera import PiCamera
from time import sleep

os.makedirs("out", exist_ok=True)

camera = PiCamera()

camera.start_preview()
sleep(5)
camera.capture("out/test.jpg")
camera.stop_preview()