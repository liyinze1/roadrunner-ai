import threading
from serial import Serial

class uart_connection:
    def __init__(self, uart='/dev/ttyS1', baudrate=230400, timeout=0.1):
        self.uart = Serial(uart, baudrate)
        
        self.receive_thread = threading.Thread(target=self.receive)
        self.receive_thread.start()
        self.ack_received = threading.Event()
        self.stop_recive = threading.Event()
        self.timeout = timeout
                
    def send_depth(self, depth):
        return self._send(b'D' + int.to_bytes(2, 2, 'big') + int.to_bytes(depth, 2, 'big'))
    
    def send_photo(self, data):
        return self._send(b'P' + int.to_bytes(len(data), 2, 'big') + data)
    
    def reqest_sleep(self):
        self.stop_recive.set()
        self.receive_thread.join()
        self._send(b'S')
        sleep_mode = self.uart.read(1)
        print('Sleep mode:', sleep_mode)
        
    def _send(self, data):
        self.uart.write(data)
        if self.ack_received.wait(timeout=self.timeout):
            self.ack_received.clear()
            return True
        else:
            print('Timeout waiting for ack')
            return False
        
    def receive(self):
        while self.stop_recive.is_set() is False:
            c = self.uart.read(1)
            print('uart get', c)
            if c == b'A':
                self.ack_received.set()
                
if __name__ == '__main__':
    uart = uart_connection()
    while True:
        depth = int(input('Enter depth: '))
        uart.send_depth(depth)
        