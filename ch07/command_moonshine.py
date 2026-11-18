from moonshine_voice import (
    ModelArch,
    AgentFlow,
    get_embedding_model,
)
import time
from gpiozero import LED, Button

running = True
button = Button(21)
led = LED(25)


def led_on(d):
    led.on()
    yield print("LED turned on.", flush=True)

def led_off(d):
    led.off()
    yield print("LED turned off.", flush=True)

def quit(d):
    global running
    running = False
    yield print("I'm glad we had this little talk.", flush=True)


# Set up the agent flow.
recogniser = (AgentFlow()
    .language("en")
    .speech(False)
    .trigger_threshold(0.6)
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
