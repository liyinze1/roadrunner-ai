import mpio
import time

gpio = mpio.GPIO(17, mpio.GPIO.OUT)
while True:
    gpio.set(True)
    time.sleep(1)
    gpio.set(False)
    time.sleep(1)