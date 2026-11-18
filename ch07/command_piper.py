from moonshine_voice import (
    AgentFlow
)
import time
from gpiozero import LED, Button
from tts_piper import TTSApp

running = True
button = Button(21)
led = LED(25)
tts_app = TTSApp("./en_US-lessac-medium.onnx")
def say(text):
    tts_app.speak(text)

def led_on(d):
    if not button.is_pressed:
        return
    led.on()
    yield say("LED turned on.")

def led_off(d):
    if not button.is_pressed:
        return
    led.off()
    yield say("LED turned off.")

def quit(d):
    if not button.is_pressed:
        return
    global running
    running = False
    yield say("I'm glad we had this little talk.")


# Set up the agent flow.
recogniser = (AgentFlow()
    .language("en")
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