#!/bin/bash

# initialize serial port
DEVICE='/dev/ttyS1'
sudo chmod 666 "$DEVICE"
stty -F "$DEVICE" 115200 cs8 -cstopb -parenb -ixon -crtscts

echo "---------------------" > "$DEVICE"

voltage=$(cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw)

echo "Voltage: $voltage" > "$DEVICE"

# initialize camera
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280

# take a photo

sleep 0.5

sudo fswebcam -i 0 -p RGB565 -r 1280x720 tinyRGB565.png

sudo -u "acme" python3 yolo_inference.py

voltage=$(cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw)

echo "Voltage: $voltage" > "$DEVICE"

echo "go to sleep" > "$DEVICE"

sudo bash sleep_modes/shutdown_with_wake.sh 344 # 10%