#!/bin/sh
echo "+$1" > /sys/class/rtc/rtc0/wakealarm
/sbin/poweroff
# /usr/bin/systemctl suspend