import threading
from serial import Serial
# import gpiod
import time


class uart_connection:
    def __init__(self, uart='/dev/ttyS1', baudrate=230400, timeout=3):
        self.uart = Serial(uart, baudrate)
        
        self.stop_recieve = threading.Event()
        self.ack_received = threading.Event()
        self.receive_thread = threading.Thread(target=self.receive)
        self.receive_thread.start()
        self.timeout = timeout
        
                
    def send_depth(self, depth):
        return self._send(b'D' + int.to_bytes(2, 2, 'big') + int.to_bytes(depth, 2, 'big'))
    
    def send_photo(self, data):
        return self._send(b'P' + int.to_bytes(len(data), 2, 'big') + data)
    
    def reqest_sleep(self):
        self.stop_recieve.set()
        self.receive_thread.join()
        self.uart.write(b'E')
        sleep_mode = self.uart.read(1)
        print('Sleep mode:', sleep_mode)
        if sleep_mode == b'S':
            # suspend to ram
            return 1
        else:
            # power off
            return 0
        
    def _send(self, data, resend=5):
        assert resend >= 0
        for _ in range(resend + 1):
            self.uart.write(data)
            if self.ack_received.wait(timeout=self.timeout):
                self.ack_received.clear()
                return True
        print('Timeout waiting for ack after', resend + 1, 'tries')
        return False
        
    def receive(self):
        while self.stop_recieve.is_set() is False:
            c = self.uart.read(1)
            print('uart get', c)
            if c == b'A':
                self.ack_received.set()
                
if __name__ == '__main__':
    uart = uart_connection()
    depth = int(input('Enter depth: '))
    uart.send_depth(depth)
    
        