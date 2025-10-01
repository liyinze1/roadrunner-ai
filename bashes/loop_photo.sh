#!/bin/bash
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1920x1080@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1920x1080 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=1080,width=1920

while true; do
    # Read GPIO line 107 on gpiochip0
    value=$(sudo gpioget gpiochip0 103) # PD7

    if [ "$value" -eq 0 ]; then
        echo "GPIO 107 is LOW → Taking photo..."

        sleep 0.5

        # Capture image with timestamp in filename
        filename="/home/acme/roadrunner-ai/photos/$(date +%Y%m%d_%H%M%S).png"
        sudo fswebcam -i 0 -p RGB565 -r 1920x1080 -S 20 --no-banner "$filename"

        echo "Saved $filename"
        
        # Optional: wait a bit before checking again
        sleep 1
    fi

    # Polling delay (adjust as needed)
    sleep 1
done
