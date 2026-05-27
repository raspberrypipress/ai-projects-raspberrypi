from time import sleep, ticks_ms
import MPU6050 # https://github.com/TimHanewich/MicroPython-Collection/tree/master/MPU6050
import machine

button = machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_UP)
i2c = machine.I2C(0, sda=machine.Pin(16), scl=machine.Pin(17))

mpu = MPU6050.MPU6050(i2c)

while True:
    accel = mpu.read_accel_data()
    print(
        ticks_ms(),
        accel[0],
        accel[1],
        accel[2],
        button.value(), 
        sep=",")
    sleep(.1)
