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


if __name__ == "__main__":

    # Load the embedding model for intent recognition.
    embedding_model_path, embedding_model_arch = get_embedding_model(
        "embeddinggemma-300m", "q4"
    )

    # Set up the intent recognizer and register some intents.
    intent_recognizer = IntentRecognizer(
        model_path=embedding_model_path,
        model_arch=embedding_model_arch,
        model_variant="q4",
        threshold=0.6,
    )
    intent_recognizer.register_intent("turn on the L E D", led_on)
    intent_recognizer.register_intent("turn off the L E D", led_off)

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
    mic_transcriber.add_listener(intent_recognizer)
    mic_transcriber.start()

    # Keep running until the user presses CTRL+C.
    print(f"CTRL+C to stop...", file=sys.stderr)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Finished.", file=sys.stderr)
    finally:
        mic_transcriber.stop()
        mic_transcriber.close()