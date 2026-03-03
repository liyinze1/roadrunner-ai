#!/bin/bash

# Check if resolution parameter is provided
RESOLUTION=${1:-1080}  # Default to 1080 if no parameter provided

sudo chmod 666 /dev/ttyS1

while true; do

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting process with ${RESOLUTION}p resolution..." >> /dev/ttyS0

    # Configure camera settings based on resolution
    if [ "$RESOLUTION" = "720" ]; then
        # 720p configuration
        sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
        sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
        sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280
        WIDTH=1280
        HEIGHT=720
    else
        # 1080p configuration (default)
        sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1920x1080@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
        sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1920x1080 field:none colorspace:srgb]'
        sudo v4l2-ctl -v pixelformat=RGBP,height=1080,width=1920
        WIDTH=1920
        HEIGHT=1080
    fi

    sleep 0.5

    echo "$(date '+%Y-%m-%d %H:%M:%S') - taking photo..." >> /dev/ttyS0

    filename="/home/acme/roadrunner-ai/photos/$(date +%Y%m%d_%H%M%S).png"
    sudo fswebcam -i 0 -p RGB565 -r ${WIDTH}x${HEIGHT} -S 20 --no-banner "$filename"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - inference..." >> /dev/ttyS0
    sudo -u "acme" python3 loop_inference.py
    code=$?   # capture exit code
    echo "sleep time $code" >> /dev/ttyS0
    
    if [ "$code" -gt 475 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - power off..." >> /dev/ttyS0
        sudo poweroff
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - standby..." >> /dev/ttyS0
        # sudo mem2io -w -i fc040018,300
        # sudo rtcwake -m mem -s 3600
        sudo rtcwake -m standby -s $code
    fi
    
done