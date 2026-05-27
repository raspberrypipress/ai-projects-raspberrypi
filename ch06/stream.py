from time import sleep, ticks_ms
import board
import busio
import adafruit_lsm303_accel
import machine

button = machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_UP)

i2c = busio.I2C(board.GP17, board.GP16)
accel = adafruit_lsm303_accel.LSM303_Accel(i2c)

while True:
    reading = accel.acceleration
    print(
        ticks_ms(),
        reading[0],
        reading[1],
        reading[2],
        button.value(), 
        sep=",")
    sleep(0.1)