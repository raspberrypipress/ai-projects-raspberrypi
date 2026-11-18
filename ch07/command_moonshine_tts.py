from moonshine_voice import (
    AgentFlow
)
import time
from gpiozero import LED

running = True
led = LED(25)

def led_on(d):
    led.on()
    yield d.say("L E D turned on.")

def led_off(d):
    led.off()
    yield d.say("L E D turned off.")

def quit(d):
    global running
    running = False
    yield d.say("I'm glad we had this little talk.")


# Set up the agent flow.
recogniser = (AgentFlow()
    .language("en")
    .speech(True)
    .trigger_threshold(0.6)
    .beeps(False)
)
recogniser.listen_for("turn on the light", led_on)
recogniser.listen_for("turn off the light", led_off)
recogniser.listen_for("quit", quit)

recogniser.load()
print('Say "quit" to stop...', flush=True)
recogniser.start_listening()
while running:
    time.sleep(0.1)

recogniser.close()