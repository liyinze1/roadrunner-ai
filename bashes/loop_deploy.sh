#!/bin/bash
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting process..." >> log.txt

sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1920x1080@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1920x1080 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=1080,width=1920
# sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
# sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
# sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280
sudo chmod 666 /dev/ttyS1

while true; do

    sleep 0.5

    echo "$(date '+%Y-%m-%d %H:%M:%S') - taking photo..." >> log.txt

    filename="/home/acme/roadrunner-ai/photos/$(date +%Y%m%d_%H%M%S).png"
    # sudo fswebcam -i 0 -p RGB565 -r 1280x720 -S 20 --no-banner "$filename"
    sudo fswebcam -i 0 -p RGB565 -r 1920x1080 -S 20 --no-banner $filename


    echo "$(date '+%Y-%m-%d %H:%M:%S') - inference..." >> log.txt
    sudo -u "acme" python3 loop.py "$filename"

    code=$?   # capture exit code

    if [ $code -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - power off..." >> log.txt
        sudo poweroff
    elif [ $code -eq 1 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - suspend to ram..." >> log.txt
        sudo mem2io -w -i fc040018,300
        sudo rtcwake -m mem -s 3600
    fi
    
done

