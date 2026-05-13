from moonshine_voice import (
    MicTranscriber,
    get_model_for_language,
    ModelArch,
    IntentRecognizer,
    get_embedding_model,
)
import time
import sys
from gpiozero import LED
from tts_piper import TTSApp

running = True
led = LED(25)
tts_app = TTSApp("./en_US-lessac-medium.onnx")
def say(text):
    mic_transcriber._should_listen = False
    tts_app.speak(text)
    mic_transcriber._should_listen = True

def led_on(trigger: str, utterance: str, similarity: float):
    led.on()
    say("LED turned on.")

def led_off(trigger: str, utterance: str, similarity: float):
    led.off()
    say("LED turned off.")

def quit(trigger: str, utterance: str, similarity: float):
    global running
    running = False
    say("I'm glad we had this little talk.")


# Load the embedding model for intent recognition.
embeddings_path, embeddings_arch = get_embedding_model(
    "embeddinggemma-300m", "q4"
)

# Set up the intent recognizer and register some intents.
recogniser = IntentRecognizer(
    model_path=embeddings_path, model_arch=embeddings_arch,
    model_variant="q4", threshold=0.6
)
recogniser.register_intent("turn on the light", led_on)
recogniser.register_intent("turn off the light", led_off)
recogniser.register_intent("quit", quit)

# Configure the transcription engine.
model_path, model_arch = get_model_for_language(
    "en", ModelArch.TINY_STREAMING
)
options = {"return_audio_data": False, 
           "identify_speakers": False,
           "vad_threshold": 0.2}
mic_transcriber = MicTranscriber(model_path=model_path,
                                 model_arch=model_arch,
                                 options=options)

# Add the recognizer to the transcriber, and start it.
mic_transcriber.add_listener(recogniser)
mic_transcriber.start()

say('Say "quit" to stop...')
while running:
    time.sleep(0.1)

mic_transcriber.stop()
mic_transcriber.close()