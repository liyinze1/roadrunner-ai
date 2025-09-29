import threading
import time
from serial import Serial

class uart_connection:
    def __init__(self, uart='/dev/ttyS1', baudrate=230400, timeout=0.1):
        self.uart = Serial(uart, baudrate)
        
    
        self.receive_thread = threading.Thread(target=self.receive)
        self.receive_thread.start()
        self.ack_received = threading.Event()
        self.timeout = timeout
                
    def send_depth(self, depth):
        return self._send(b'D' + int.to_bytes(2, 2, 'big') + int.to_bytes(depth, 2, 'big'))
        
    def _send(self, data):
        self.uart.write(data)
        if self.ack_received.wait(timeout=self.timeout):
            self.ack_received.clear()
            print('Ack received')
            return True
        else:
            print('Timeout waiting for ack')
            return False
        
    def receive(self):
        while True:
            c = self.uart.read(1)
            print('uart get', c)
            if c == b'A':
                self.ack_received.set()
                
                
if __name__ == '__main__':
    uart = uart_connection()
    while True:
        try:
            depth = int(input('Enter depth: '))
            uart.send_depth(depth)
        except:
            exit(1)
        