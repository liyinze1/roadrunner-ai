import threading
from serial import Serial
# import gpiod
import time


class uart_connection:
    def __init__(self, uart='/dev/ttyS1', baudrate=230400, timeout=3, retry=5):
        self.uart = Serial(uart, baudrate, timeout=timeout)        
        self.retry = retry
                
    def send_depth(self, depth):
        for _ in range(self.retry):
            result = self._send(b'D' + int.to_bytes(2, 2, 'big') + int.to_bytes(depth, 2, 'big'))
            if result == b'A':
                return True
        return False
    
    def send_photo(self, data):
        for _ in range(self.retry):
            result = self._send(b'P' + int.to_bytes(len(data), 2, 'big') + data)
            if result == b'A':
                return True
        return False
    
    def reqest_sleep(self):
        for _ in range(self.retry):
            sleep_mode = self._send(b'E')
            if sleep_mode == b'S':
                # suspend to ram
                return 1
            elif sleep_mode == b'P':
                # power off
                return 0
        return 0  # default to power off
        
    def _send(self, data):
        self.uart.write(data)
        c = self.uart.read(1)
        print('get response:', c)
        return c
        
                
if __name__ == '__main__':
    uart = uart_connection()
    depth = int(input('Enter depth: '))
    uart.send_depth(depth)
    
        