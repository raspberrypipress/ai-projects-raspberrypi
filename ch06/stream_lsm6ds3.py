from time import sleep, ticks_ms
from lsm6ds3 import LSM6DS3, NORMAL_MODE_104HZ # https://github.com/pimoroni/lsm6ds3-micropythonv
import machine

button = machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_UP)
i2c = machine.I2C(0, sda=machine.Pin(16), scl=machine.Pin(17))

sensor = LSM6DS3(i2c, mode=NORMAL_MODE_104HZ, address=0x6b)

while True:
    accel = sensor.get_readings()
    print(
        ticks_ms(),
        accel[0],
        accel[1],
        accel[2],
        button.value(), 
        sep=",")
    sleep(.1)

