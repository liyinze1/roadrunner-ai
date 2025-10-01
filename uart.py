import threading
from serial import Serial
import gpiod
import time




class uart_connection:
    def __init__(self, uart='/dev/ttyS1', baudrate=230400, timeout=0.1):
        self.uart = Serial(uart, baudrate)
        
        self.stop_recieve = threading.Event()
        self.ack_received = threading.Event()
        self.receive_thread = threading.Thread(target=self.receive)
        self.receive_thread.start()
        self.timeout = timeout
        
        self.wake_gpio()
        
    def wake_gpio(self, gpio_pin=103):
        chip = gpiod.Chip('gpiochip0')
        line = chip.get_line(gpio_pin)
        line.request(consumer='my-app', type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
        
        while True:
            line.set_value(0)
            time.sleep(0.1)
            line.set_value(1)
            if self.ack_received.wait(timeout=1):
                self.ack_received.clear()
                print('Wake up success')
                break
        line.release()
                
    def send_depth(self, depth):
        return self._send(b'D' + int.to_bytes(2, 2, 'big') + int.to_bytes(depth, 2, 'big'))
    
    def send_photo(self, data):
        return self._send(b'P' + int.to_bytes(len(data), 2, 'big') + data)
    
    def reqest_sleep(self):
        self.stop_recieve.set()
        self.receive_thread.join()
        self._send(b'S')
        sleep_mode = self.uart.read(1, timeout=1)
        print('Sleep mode:', sleep_mode)
        if sleep_mode == b'R':
            return 1
        else:
            return 0
        
    def _send(self, data):
        self.uart.write(data)
        if self.ack_received.wait(timeout=self.timeout):
            self.ack_received.clear()
            return True
        else:
            print('Timeout waiting for ack')
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
    
        