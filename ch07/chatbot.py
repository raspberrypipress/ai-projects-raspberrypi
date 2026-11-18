from moonshine_voice import (
    MicTranscriber,
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

class TextListener(TranscriptEventListener):
    def on_line_completed(self, event):
        if not button.is_pressed:
            diags.print(f"Skipping: {event.line.text}")
            return

        led.on()
        diags.print(f"Transcribed: {event.line.text}")
        response = llm.generate(event.line.text)
        tts_app.speak(response)
        led.off()

# Configure the transcriber and specify the model for the 
# language we want to transcribe
options = {"return_audio_data": False, 
           "identify_speakers": False,
           "vad_threshold": 0.2}
mic_transcriber = (MicTranscriber()
                   .language("en")
                   .model_arch(ModelArch.TINY_STREAMING)
                   .options(options)
)

# Add the listener to the transcriber and start it.
mic_transcriber.add_listener(TextListener())
mic_transcriber.load()
mic_transcriber.start()

diags.print("Ready! CTRL+C to stop...")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    diags.print("Finished.")
finally:
    mic_transcriber.stop()