# 640x480_RGB565
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:RGB565_2X8_LE/640x480@1/24 field:none colorspace:srgb]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:RGB565_2X8_LE/640x480 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=480,width=640
sudo fswebcam -p RGB565 -r 640x480 -S 20 RGB565_LE_640_480.png

# 1280x720_SBGGR8
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1280x720@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1280x720 field:none colorspace:srgb]'
sudo v4l2-ctl -v pixelformat=RGBP,height=720,width=1280
sudo fswebcam -i 0 -p RGB565 -r 1280x720 1280x720_SBGGR8.png

# 1920x1080
sudo media-ctl -d /dev/media0 --set-v4l2 '4:0[fmt:SBGGR8_1X8/1920x1080@1/30 field:none colorspace:srgb xfer:srgb ycbcr:601 quantization:full-range]'
sudo media-ctl -d /dev/media0 --set-v4l2 '"atmel_isc_scaler":0[fmt:SBGGR8_1X8/1920x1080 field:none colorspace:srgb]'

sudo v4l2-ctl -v pixelformat=RGBP,height=1080,width=1920
sudo fswebcam -p RGB565 -r 1920x1080 -S 20 fullRGB565.png

sudo v4l2-ctl -v pixelformat=AR24,height=1080,width=1920
sudo fswebcam -p ABGR32 -r 1920x1080 -S 20 fullABGR32.png

sudo v4l2-ctl -v pixelformat=GREY,height=1080,width=1920
sudo fswebcam -p GREY -r 1920x1080 -S 20 fullGREY.png

sudo v4l2-ctl -v pixelformat=YU12,height=1080,width=1920
sudo fswebcam -p YUV420P -r 1920x1080 -S 20 fullYUV420P.png