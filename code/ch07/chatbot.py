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
speaking = False
def say(text):
    global speaking
    speaking = True
    tts_app.speak(text)
    speaking = False

# FIXME: add a button. Ignore lines unless the button is pressed.
class FileListener(TranscriptEventListener):
    def __init__(self):
        self.ignore_lines = []
        super().__init__()

    def on_line_started(self, event):
        if speaking:
            self.ignore_lines.append(event.line.line_id)

    def on_line_completed(self, event):
        if event.line.line_id in self.ignore_lines:
            self.ignore_lines.remove(event.line.line_id)
            return

        diags.print(f"Transcribed: {event.line.text}")
        response = llm.generate(event.line.text)
        say(response)

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