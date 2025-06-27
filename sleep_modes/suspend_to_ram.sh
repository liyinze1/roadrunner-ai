sudo mem2io -w -i fc040018,300
sudo rtcwake -m mem -s "$1"
echo "Woke up from suspend to RAM after $1 seconds"