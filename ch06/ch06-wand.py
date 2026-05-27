import emlearn_trees
import array
import gc
import _thread
from time import sleep, ticks_ms
import board
import busio
import adafruit_lsm303_accel
import adafruit_lis2mdl
import neopixel
from led_helpers import hsv_to_rgb, sparkle

multiplier = 1000
run_sparkle = False
np = neopixel.NeoPixel(machine.Pin(2), 10)

def core1_loop():
    global run_sparkle, np
    while True:
        while not run_sparkle:
            sleep(0.1)
        sparkle(np,10,7,3,5,0.9)
        run_sparkle = False

_thread.start_new_thread(core1_loop, ())

i2c = busio.I2C(board.GP17, board.GP16)
mag = adafruit_lis2mdl.LIS2MDL(i2c)
accel = adafruit_lsm303_accel.LSM303_Accel(i2c)

print("loading model")
model = emlearn_trees.new(300, 10000, 200)

with open('flick_model.csv', 'r') as f:
    emlearn_trees.load_model(model, f)

resout = array.array('f',[0,0])
window = [0] * 30

print("running")
while True:
    del window[:3] # Remove the first 3 elements
    reading = accel.acceleration
    window.append(reading[0] * multiplier)
    window.append(reading[1] * multiplier)
    window.append(reading[2] * multiplier)
    model.predict(array.array('h', window), resout)
    if(resout[1] > 0.60):
        print(f"flick detected at {ticks_ms()} ",
              f"{resout[1]}% certainty")
        run_sparkle = True
        # Clear the window to avoid multiple detections
        window = [0] * 30
    sleep(0.1)