from moonshine_voice import (
    AgentFlow,
    TextToSpeech
)
import time
import sys
from gpiozero import LED, Button
import sounddevice as sd
import numpy as np

running = True
button = Button(21)
led = LED(25)
tts = TextToSpeech().language("en_us")
tts.load()
tts_cache = {}

# This version improves performance by caching synthesised audio.
def say(d, text):
    global tts_cache
    if text not in tts_cache:
        print(f"Synthesizing: {text}")
        audio, sample_rate = tts.synthesize(text)
        silence = np.zeros(int(0.5 * sample_rate))
        audio = np.concatenate([silence, audio])
        tts_cache[text] = {"audio": audio,
                           "rate": sample_rate}

    sd.play(tts_cache[text]["audio"],
            tts_cache[text]["rate"])
    sd.wait()

def led_on(d):
    led.on()
    yield say(d, "L E D turned on.")

def led_off(d):
    led.off()
    yield say(d, "L E D turned off.")

def quit(d):
    global running
    running = False
    yield say(d, "I'm glad we had this little talk.")


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