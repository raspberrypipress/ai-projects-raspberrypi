import signal
import sys
from time import time
from piper import PiperVoice
import numpy as np
import sounddevice as sd

voice = PiperVoice.load("./en_US-lessac-medium.onnx")
stream = sd.OutputStream(samplerate=voice.config.sample_rate,
                         channels=1, dtype='int16')
stream.start()
time.sleep(0.1)  # Give the stream a moment to start

signal.signal(signal.SIGINT, signal.SIG_IGN)
print(f"Ready.", file=sys.stderr)
for text in sys.stdin:
    for chunk in voice.synthesize(text):
        int_data = np.frombuffer(chunk.audio_int16_bytes,
                                 dtype=np.int16)
        stream.write(int_data)

stream.stop()
stream.close()
print(f"Text-to-speech completed.", file=sys.stderr)
