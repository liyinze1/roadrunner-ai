import water_inference
import uart
import sys

arg = sys.argv
if len(arg) > 1:
    image_path = arg[0]
else:
    image_path = "water.jpg"
    
    
print("Image path:", image_path)    
depth = water_inference.main(image_path)

uart_conn = uart.uart_connection()
uart_conn.send_depth(depth)
sleep_mode = uart_conn.reqest_sleep()

exit(sleep_mode)