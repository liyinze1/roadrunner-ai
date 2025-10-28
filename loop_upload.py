import water_inference
import uart
import os


uart_conn = uart.uart_connection()

photo_dir = '/photos'
image_path = 'water.png'
payload_length = 500

if os.path.exists(photo_dir):
    image_files = sorted(os.listdir(photo_dir))
    if len(image_files) > 0:
        image_path = os.path.join(photo_dir, image_files[-1])
f = open(image_path, 'rb')
data = f.read()
f.close()
i = 0
sn = 0
while i < len(data):
    chunk = int.to_bytes(sn, 2, 'big') + data[i:min(i+payload_length, len(data))]
    uart_conn.send_photo(chunk)
    i += payload_length
    sn += 1

sleep_mode = uart_conn.reqest_sleep()

exit(sleep_mode)