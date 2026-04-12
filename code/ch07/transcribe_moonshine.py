from moonshine_voice.mic_transcriber import MicTranscriber
from moonshine_voice import get_model_for_language, ModelArch
from moonshine_voice.transcriber import TranscriptEventListener
import time
import sys

class FileListener(TranscriptEventListener):
    def on_line_completed(self, event):
        print(event.line.text)


if __name__ == "__main__":

    lang = "en"
    model = ModelArch.TINY
    model_path, model_arch = get_model_for_language(
        wanted_language=lang, wanted_model_arch=model
    )

    options = {"return_audio_data": False, 
               "identify_speakers": False,
               "transcription_interval": 2.0}
    mic_transcriber = MicTranscriber(model_path=model_path,
                                     model_arch=model_arch,
                                     options=options)

    listener = FileListener()
    mic_transcriber.add_listener(listener)

    print(f"Ctrl+C to stop...", file=sys.stderr)
    mic_transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Finished!", file=sys.stderr)
    finally:
        mic_transcriber.stop()
        mic_transcriber.close()