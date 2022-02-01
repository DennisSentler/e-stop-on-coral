import pyaudio
import numpy as np
import threading
import measure
import python_speech_features as speech_features

class AudioProvider(threading.Thread):
    def __init__(
            self, sample_rate=16000, 
            format=pyaudio.paInt16, 
            reading_chunk=1000,
            audio_window_s=1
        ):
        threading.Thread.__init__(self)
        self.daemon=True
        self._lock = threading.Lock()
        self._format = format
        self._sample_rate = sample_rate
        self._output_in_s = audio_window_s
        self._audio_window = np.zeros(audio_window_s * sample_rate, dtype=np.int16)
        self._chunk = reading_chunk
        self._running = False
        self.clock = measure.InferenceClock(name="AudioProvider")
        p = pyaudio.PyAudio()
        self._stream = p.open(format=format,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=reading_chunk)

    def _bytes_to_array(self, bytes):
        return np.frombuffer(bytes, np.int16)

    def stop(self):
        self._running = False
        self._stream.stop_stream()

    def read_audio_window(self):
        return self._audio_window

    def run(self):
        self._stream.start_stream()
        self._running = True
        while self._running:
            self.clock.start()
            audio_bytes = self._stream.read(self._chunk)
            audio_arr = self._bytes_to_array(audio_bytes)
            tmp_audio = self._audio_window
            with self._lock:
                tmp_audio = [*tmp_audio, *audio_arr]
                del tmp_audio[:self._chunk]
                self._audio_window = tmp_audio
                self.clock.stop()
        self._stream.stop_stream()

def pre_proc_audio(audio, sample_rate=16000, windows_size=640, window_stride=320, num_mfcc=10):
    audio = np.interp(audio, (np.iinfo(np.int16).min, np.iinfo(np.int16).max), (-1, +1))
    audio = audio.astype(np.float32)

    window_size_in_seconds = windows_size/sample_rate
    window_stride_in_seconds = window_stride/sample_rate
    mfccs = speech_features.mfcc(
        audio,
        samplerate=sample_rate,
        winlen=window_size_in_seconds,
        winstep=window_stride_in_seconds,
        numcep=num_mfcc,
        nfilt=40,
        lowfreq=20,
        highfreq=4000)
    mfccs = mfccs.astype(np.float32)
    mfccs = np.expand_dims(mfccs, axis=0)
    return mfccs