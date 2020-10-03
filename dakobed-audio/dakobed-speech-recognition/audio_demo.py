"""PyAudio Example: Play a WAVE file."""
import pyaudio
import wave
import sys

CHUNK = 1024

audio_file = 'data/train/audio/bird/0a9f9af7_nohash_0.wav'


wf = wave.open(audio_file, 'rb')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

data = wf.readframes(CHUNK)

while data != '':
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()
