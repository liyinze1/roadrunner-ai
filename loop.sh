# initialize camera
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280

# take a photo
sleep 0.5
sudo fswebcam -i 0 -p RGB565 -r 1280x720 tinyRGB565.png

# inference
sudo -u "acme" python3 water_inference.py

# suspend to ram
echo "finished inference, suspending now..."
sudo bash sleep_modes/suspend_to_ram.sh 9999

# wake up here
echo "wake up from suspend"

# take a photo
sleep 0.5
sudo fswebcam -i 0 -p RGB565 -r 1280x720 tinyRGB565.png

# inference
sudo -u "acme" python3 water_inference.py

# power off
echo "going to shutdown now..."
sudo poweroff