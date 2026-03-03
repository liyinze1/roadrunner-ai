# 640x480_RGB565
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:RGB565_2X8_LE/640x480@1/24 field:none colorspace:srgb]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:RGB565_2X8_LE/640x480 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=480,width=640
sudo fswebcam -p RGB565 -r 640x480 -S 20 RGB565_LE_640_480.png

# 1280x720_SBGGR8
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280
sudo fswebcam -i 0 -p RGB565 -r 1280x720 -S 20 1280x720_SBGGR8.png
# sudo fswebcam -r 1280x720 -p RGB565  --set brightness=-1023 --set contrast=512 -S 20 test1.jpg

# 1920x1080
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SRGGB8_1X8/1920x1080@1/30 field:none colorspace:srgb]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SRGGB8_1X8/1920x1080 field:none colorspace:srgb]'
sudo v4l2-ctl -d /dev/video0 -v pixelformat=RGBP,height=1080,width=1920

sudo fswebcam -d /dev/video0 -i 0 -p BAYER -r 1920x1080 full.png

sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1920x1080@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1920x1080 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=1080,width=1920
sudo v4l2-ctl  --set-parm=15
sudo fswebcam -i 0 -p RGB565 -r 1920x1080 -S 20 --no-banner 1920x1080_SBGGR8.png

# Check if this file exists
cat /sys/devices/platform/ahb/ahb:apb/fc028000.i2c/power/control

# If yes, try this:
echo on | sudo tee /sys/devices/platform/ahb/ahb:apb/fc028000.i2c/power/control
echo auto | sudo tee /sys/devices/platform/ahb/ahb:apb/fc028000.i2c/power/control