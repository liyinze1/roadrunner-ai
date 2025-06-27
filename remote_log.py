import serial
import time

def main():
    serial_port = '/dev/ttyS0'  # Change this to your actual serial device
    baud_rate = 115200              # Adjust as needed
    log_file = 'log.txt'

    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser, open(log_file, 'a') as log:
            print(f'Logging started. Reading from {serial_port} at {baud_rate} baud.')
            while True:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='replace').strip()
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    log_line = f'[{timestamp}] {line}\n'
                    log.write(log_line)
                    log.flush()
                    print(log_line, end='')

    except serial.SerialException as e:
        print(f'Serial error: {e}')
    except KeyboardInterrupt:
        print('\nLogging stopped by user.')

if __name__ == '__main__':
    main()
