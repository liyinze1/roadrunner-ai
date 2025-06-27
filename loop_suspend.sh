#!/bin/bash

# --- Setup (run only once) ---
# For more camera setup, refer to https://github.com/linux4sam/video-capture-at91/tree/master/sama5d2/ov5640
echo Preparing OV5640 in RAW BAYER MODE
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280
echo Ready to capture at 1280x720

# --- Loop ---
sleep 2

echo 'Starting loop...'
start_time=$(date +%s.%N)
iteration=0

while true; do
    ((iteration++))
    
    printf "Iteration: %d\n" "$iteration"
    printf "Iteration: %d\n" "$iteration" >> log_10.txt

    sudo fswebcam -i 0 -p RGB565 -r 1280x720 tinyRGB565.png

    # Run Python script as original user
    sudo -u "acme" python3 yolo_inference.py


    end_time=$(date +%s.%N)
    elapsed=$(awk "BEGIN {print $end_time - $start_time}")

    voltage=$(cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw)


    printf "Time elapsed: %.3f seconds | Voltage: %s\n" "$elapsed" "$voltage"
    # Write to log file
    printf "Time elapsed: %.3f seconds | Voltage: %s\n" "$elapsed" "$voltage" >> log_10.txt

    sudo bash sleep_modes/suspend_to_ram.sh 346

    end_time=$(date +%s.%N)
    elapsed=$(awk "BEGIN {print $end_time - $start_time}")
    voltage=$(cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw)

    printf "wake up: %.3f seconds | Voltage: %s\n" "$elapsed" "$voltage"
    # Write to log file
    printf "wake up: %.3f seconds | Voltage: %s\n" "$elapsed" "$voltage" >> log_10.txt

done
