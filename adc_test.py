import time

def get_adc(id):
    with open("/sys/bus/iio/devices/iio:device0/in_voltage%d_raw" % id) as f:
        value=int(f.read())
        return value

start = time.time()

f = open('log.txt', 'w')
f.write('voltage, time\n')
f.close()

while True:
    # adc = ADC(0)
    voltage = get_adc(0)
    f = open('log.txt', 'a')
    f.write('%d, %.2f' % (voltage, time.time() - start))
    # f.write('%.2f' % (time.time() - start))
    f.write('\n')
    f.close()
    time.sleep(10)
    