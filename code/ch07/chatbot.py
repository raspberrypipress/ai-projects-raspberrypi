from moonshine_voice import (
    MicTranscriber,
    get_model_for_language,
    ModelArch,
    TranscriptEventListener,
)
import time
from llm import LLMApp
from tts_piper import TTSApp
from rich.console import Console
from gpiozero import LED, Button

button = Button(21)
led = LED(25)

# Set up the LLM and TTS apps.
diags = Console(stderr=True, style="purple")
llm = LLMApp("./Qwen2-1.5B-Instruct.hef", diags,
             "You are a helpful assistant.")
tts_app = TTSApp("./en_US-lessac-medium.onnx")

def say(text):
    tts_app.speak(text)

class FileListener(TranscriptEventListener):
    def on_line_completed(self, event):
        if button.is_pressed:
            led.on()
            diags.print(f"Transcribed: {event.line.text}")
            response = llm.generate(event.line.text)
            say(response)
            led.off()
        else:
            diags.print(f"Skipping: {event.line.text}")

# Load the model and set up the transcriber.
model_path, model_arch = get_model_for_language(
    "en", ModelArch.TINY_STREAMING
)
options = {"return_audio_data": False, 
           "identify_speakers": False,
           "vad_threshold": 0.2}
mic_transcriber = MicTranscriber(model_path=model_path,
                                 model_arch=model_arch,
                                 options=options)
mic_transcriber.add_listener(FileListener())
mic_transcriber.start()

diags.print("CTRL+C to stop...")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    diags.print("Finished.")
finally:
    mic_transcriber.stop()
    mic_transcriber.close()