#!/bin/bash
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1920x1080@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1920x1080 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=1080,width=1920
sudo chmod 666 /dev/ttyS1
sudo chmod 666 /dev/gpiochip0

while true; do

    filename="/home/acme/roadrunner-ai/photos/$(date +%Y%m%d_%H%M%S).png"
    sudo fswebcam -i 0 -p RGB565 -r 1920x1080 -S 20 --no-banner "$filename"

    # sudo -u "acme" python3 loop.py "$filename"

    sudo -u "acme" python3 water_inference.py

    sudo poweroff

    code=$?   # capture exit code

    if [ $code -eq 0 ]; then
        echo "shutdown"
        sudo poweroff
    elif [ $code -eq 1 ]; then
        echo "suspend"
        sudo mem2io -w -i fc040018,300
        sudo rtcwake -m mem -s 3600
    fi
    
done






python3 test.py
echo "Exit code: $?"
