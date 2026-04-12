# First, make a venv, activate it, then do:
#    pip install moonshine-voice transformers nltk torch sentencepiece sacremoses
# Run `apt install libportaudio2 pavucontrol pulseaudio-utils` beforehand, then
# follow these steps to capture system audio
# https://unix.stackexchange.com/questions/82259/how-to-pipe-audio-output-to-mic-input/82297#82297

from moonshine_voice.mic_transcriber import MicTranscriber
from moonshine_voice import get_model_for_language, ModelArch
from moonshine_voice.transcriber import TranscriptEventListener
from transformers import MarianMTModel, MarianTokenizer
import time
import sys


class FileListener(TranscriptEventListener):
    def __init__(self, mt_model, tokenizer):
        super().__init__()
        self.mt_model = mt_model
        self.tokenizer = tokenizer

    def on_line_completed(self, event):
        print(event.line.text)
        inputs = self.tokenizer(event.line.text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True)
        translated = self.mt_model.generate(**inputs)
        r = self.tokenizer.decode(translated, 
                                  skip_special_tokens=True)
        print(r[0])


if __name__ == "__main__":

    lang = "en"
    stt_model = ModelArch.TINY_STREAMING
    stt_model_path, stt_model_arch = get_model_for_language(
        wanted_language=lang, wanted_model_arch=stt_model
    )

    # TODO: move the mt stuff into a class, and pass an object to the listener instead of using variables
    # TODO: adopt a multiprocessing model so we can use one core for transcription and the other for translation
    mt_model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    mt_model = MarianMTModel.from_pretrained(mt_model_name)

    options = {"return_audio_data": False}
    mic_transcriber = MicTranscriber(model_path=stt_model_path,
                                     model_arch=stt_model_arch,
                                     blocksize=4096,
                                     options=options)

    listener = FileListener(mt_model=mt_model, 
                            tokenizer=tokenizer)
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
