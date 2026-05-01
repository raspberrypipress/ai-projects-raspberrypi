# pip install piper-tts sounddevice
# python3 -m piper.download_voices en_US-lessac-medium
import wave
import numpy as np
from piper import PiperVoice
import sounddevice as sd

voice = PiperVoice.load("./en_US-lessac-medium.onnx")
stream = sd.OutputStream(samplerate=voice.config.sample_rate,
                         channels=1, dtype='int16')
stream.start()

text = """Since 2012, we’ve been designing single-board and
modular computers, built on the Arm architecture, and
running the Linux operating system. Whether you’re an
educator looking to excite the next generation of computer
scientists; an enthusiast searching for inspiration for your
next project; or an OEM who needs a proven rock-solid
foundation for your next generation of smart products,
there’s a Raspberry Pi computer for you."""

for chunk in voice.synthesize(text):
    int_data = np.frombuffer(chunk.audio_int16_bytes,
                             dtype=np.int16)
    stream.write(int_data)

stream.stop()
stream.close()
