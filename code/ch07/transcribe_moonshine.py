from moonshine_voice import (
    MicTranscriber,
    get_model_for_language,
    ModelArch,
    TranscriptEventListener,
)
import time
import sys

# Define a listener to display completed lines of transcription.
class FileListener(TranscriptEventListener):
    def on_line_completed(self, event):
        print(event.line.text)


if __name__ == "__main__":

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

    # Add the listener to the transcriber, and start it.
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