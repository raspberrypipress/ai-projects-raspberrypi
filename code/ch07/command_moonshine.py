from moonshine_voice import (
    IntentRecognizer,
    MicTranscriber,
    ModelArch,
    get_embedding_model,
    get_model_for_language,
)
import time
import sys
from gpiozero import LED

led = LED(3)

def led_on(trigger: str, utterance: str, similarity: float):
    led.on()
    print("LED turned on.")

def led_off(trigger: str, utterance: str, similarity: float):
    led.off()
    print("LED turned off.")

def quit(trigger: str, utterance: str, similarity: float):
    global running
    running = False


running = True
if __name__ == "__main__":

    # Load the embedding model for intent recognition.
    embeddings_path, embeddings_arch = get_embedding_model(
        "embeddinggemma-300m", "q4"
    )

    # Set up the intent recognizer and register some intents.
    recogniser = IntentRecognizer(
        model_path=embeddings_path,
        model_arch=embeddings_arch,
        model_variant="q4",
        threshold=0.6,
    )
    recogniser.register_intent("turn on the L E D", led_on)
    recogniser.register_intent("turn off the L E D", led_off)
    recogniser.register_intent("quit", quit)

    # Configure the transcription engine.
    model_path, model_arch = get_model_for_language(
        "en", ModelArch.TINY_STREAMING
    )
    options = {"return_audio_data": False, 
               "identify_speakers": False}
    mic_transcriber = MicTranscriber(model_path=model_path,
                                     model_arch=model_arch,
                                     options=options)

    # Add the recognizer to the transcriber, and start it.
    mic_transcriber.add_listener(recogniser)
    mic_transcriber.start()

    # Keep running until the user presses CTRL+C.
    print('Say "quit" to stop...', file=sys.stderr)
    while running:
        time.sleep(0.1)
        print(".", end="", flush=True)

    mic_transcriber.stop()
    mic_transcriber.close()