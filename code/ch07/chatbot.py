from moonshine_voice import (
    MicTranscriber,
    get_model_for_language,
    ModelArch,
    TranscriptEventListener,
)
import time
import sys
from llm import LLMApp
from tts_piper import TTSApp
from rich.console import Console

# Define a listener to display completed lines of transcription.
diags = Console(stderr=True, style="purple")
llm = LLMApp("Qwen2-1.5B-Instruct", diags,
             "You are a helpful assistant.")

tts_app = TTSApp("./en_US-lessac-medium.onnx")
def say(text):
    mic_transcriber._should_listen = False
    tts_app.speak(text)
    mic_transcriber._should_listen = True

class FileListener(TranscriptEventListener):
    def on_line_completed(self, event):
        say(llm.generate(event.line.text))


# Load the model for the language we want to transcribe.
model_path, model_arch = get_model_for_language(
    "en", ModelArch.TINY_STREAMING
)

# Configure the transcriber.
options = {"return_audio_data": False, 
           "identify_speakers": False}
mic_transcriber = MicTranscriber(model_path=model_path,
                                 model_arch=model_arch,
                                 options=options)

# Add the listener to the transcriber and start it.
mic_transcriber.add_listener(FileListener())
mic_transcriber.start()

# Keep running until the user presses CTRL+C.
print("CTRL+C to stop...", file=sys.stderr)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Finished.", file=sys.stderr)
finally:
    mic_transcriber.stop()
    mic_transcriber.close()