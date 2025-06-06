import mpio
import serial
import yaml

class Flute_Connection:
    def __init__(self):
        
        self.s1 = mpio.GPIO(11, mpio.GPIO.OUT) # PA11
        self.s2 = mpio.GPIO(13, mpio.GPIO.OUT) # PA13
        self.s3 = mpio.GPIO(17, mpio.GPIO.OUT) # PA17
        
        self.s1.set(True)
        self.s2.set(True)
        self.s3.set(True)
    
    def read_pv_voltage(self):
        with open('/sys/bus/iio/devices/iio:device0/in_voltage0_raw') as f:
            value=int(f.read())
        return value
    
    def read_cap_voltage(self):
        with open('/sys/bus/iio/devices/iio:device0/in_voltage1_raw') as f:
            value=int(f.read())
        return value
    

class Modem_Connection:
    def __init__(self):
        
        
        pass
    
class RoadRunner:
    
    @staticmethod
    def suspend_to_ram():
        pass
    
    @staticmethod
    def shutdown():
        pass
    
    @staticmethod
    def standby():
        pass
    
class 