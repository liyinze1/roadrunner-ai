### Roadrunner Pinout

#### 1. UART

|Signal| Pin | Pin on nRF |
|-|-|-|
| RX | PC12 | TX | 
| TX | PC13 | RX |
| wkup | wkup | A1/14 |

#### 2. Camera

https://www.acmesystems.it/roadrunner_isc

#### 3. GPIO

Give permission

```
sudo chmod 666 /dev/ttyS1
sudo chmod 666 /dev/gpiochip0
sudo gpioinfo gpiochip0
```


### Run the loop service

```bash
sudo cp loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable loop.service
sudo systemctl start loop.service
sudo systemctl status loop.service
sudo systemctl stop loop.service
sudo systemctl disable loop.service
```

```bash
sudo systemctl stop zerotier-one
sudo systemctl disable zerotier-one

sudo systemctl enable zerotier-one
sudo systemctl start zerotier-one
```