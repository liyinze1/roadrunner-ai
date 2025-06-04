### Roadrunner Pinout

#### 1. ADC

|Signal|Pin|Comment|
|-|-|-|
| ADC0  | PD19  | Photovoltaic
| ADC1  | PD20  | Super Cap Volatage
| ADC2  | PD21  |
| ADC3  | PD22  |
| ADC4  | PD23  |
| ADC5  | PD24  |
| ADC6  | PD25  |
| ADC7  | PD26  |
| ADC8  | PD27  |
| ADC9  | PD28  |
| ADC10 | PD29  |
| ADC11 | PD30  |

#### 2. UART

|Signal| Pin|
|-|-|
| RX | PC12 |
| TX | PC13 |

#### 3. Camera

https://www.acmesystems.it/roadrunner_isc

#### 4. GPIO

Give permission

```
sudo chmod 666 /dev/gpiochip0
sudo gpioinfo gpiochip0
```
|Pin|Switch|Comment|
|-|-|-|
| PA11  | S1  | Photovoltaic Measurement
| PA13  | S2  | Open-circuit Feature
| PA17  | S3  | Sensor Power Switch
