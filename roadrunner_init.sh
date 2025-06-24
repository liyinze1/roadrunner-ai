sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip git v4l-utils fswebcam
sudo apt install -y libjpeg-dev zlib1g-dev # For Pillow
sudo apt install -y libgl1-mesa-glx libgl1-mesa-dri # For OpenCV
sudo apt install -y libopencv-dev python3-opencv # For OpenCV Python bindings
curl -s https://install.zerotier.com | sudo bash