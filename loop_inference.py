import uart
import water_inference_unet

depth = water_inference_unet.main()

uart_conn = uart.uart_connection()
uart_conn.send_depth(depth)
sleep_mode = uart_conn.reqest_sleep()

exit(sleep_mode)