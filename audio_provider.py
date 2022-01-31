import pyaudio
import numpy as np
from threading import Thread
import measure

class AudioProvider(threading.Thread):
    def __init__(
            self, sample_rate=16000, 
            format=pyaudio.paInt16, 
            channels=1, 
            reading_chunk=1000,
            audio_window_s=1
        ):
        threading.Thread.__init__(self)
        self._lock = threading.Lock()
        self._format = format
        self._sample_rate = sample_rate
        self._output_in_s = audio_window_s
        self._audio_window = np.zeros(_output_in_s * sample_rate)
        self._chunk = reading_chunk
        self._running = False
        self.clock = measure.InferenceClock(name="AudioProvider")
        p = pyaudio.PyAudio()
        self._stream = p.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=reading_chunk)

    def _bytesToArray(bytes):
        return np.frombuffer(bytes, np.int16)

    def terminate():
        self._running = False
        self._stream.stop_stream()
        self._stream.close()

    def readAudioWindow():
        return self._audio_window

    def run(self):
        self._stream.start_stream()
        self._running = True
        while self._running:
            self.clock.start()
            audio_bytes = _bytesToArray(self._stream.read(self._chunk))
            tmp_audio = self._audio_window
            with self._lock:
                tmp_audio.extend(audio_bytes)
                del tmp_audio[:self._chunk]
                self._audio_window = tmp_audio
                self.clock.stop()
        self._stream.stop_stream()