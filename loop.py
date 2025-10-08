import water_inference
import uart


depth = water_inference.main()

uart_conn = uart.uart_connection()
uart_conn.send_depth(depth)
sleep_mode = uart_conn.reqest_sleep()

exit(sleep_mode)