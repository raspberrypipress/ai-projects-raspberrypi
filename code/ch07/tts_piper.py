import signal
import sys
import time
from piper import PiperVoice
import numpy as np
import sounddevice as sd

class TTSApp:

    def __init__(self, model):
        self.voice = PiperVoice.load(model)
        sample_rate = self.voice.config.sample_rate
        self.stream = sd.OutputStream(samplerate=sample_rate,
                                     channels=1, dtype='int16')
        self.stream.start()
        time.sleep(0.5)  # Give the stream a moment to start

    def speak(self, text):
        for chunk in self.voice.synthesize(text):
            int_data = np.frombuffer(chunk.audio_int16_bytes,
                                     dtype=np.int16)
        self.stream.write(int_data)
    
    def __del__(self):
        self.stream.stop()
        self.stream.close()

if __name__ == "__main__":
    app = TTSApp("./en_US-lessac-medium.onnx")

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"Ready.", file=sys.stderr)
    for text in sys.stdin:
        app.speak(text.strip())

    print(f"Text-to-speech completed.", file=sys.stderr)
