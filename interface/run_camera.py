import time

from display import Display

display = Display(modes = ["mode one", "mode two", "mode three", "mode four", "mode five"])
while True:
    display.read_buttons()
    display.draw()
    time.sleep(0.01)
