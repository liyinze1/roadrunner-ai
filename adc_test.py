from mpio import ADC
import time

while True:
    adc = ADC(0)
    print("ADC value: %d" % adc.value(0))
    time.sleep(1)