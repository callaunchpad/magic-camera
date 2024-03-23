import time

from display import Display

display = Display(modes = ["mode one", "mode two"])
while True:
    display.read_buttons()
    display.draw()
    time.sleep(0.01)
