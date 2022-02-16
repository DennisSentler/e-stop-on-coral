import pyaudio
import numpy as np
import threading
import python_speech_features as speech_features
import contextlib
import os, sys
from logger import log
from config import AUDIO, MFCC

class AudioProvider(threading.Thread):
    def __init__(
            self
        ):
        threading.Thread.__init__(self)
        self.daemon=True
        self._lock = threading.Lock()
        self._audio_window = np.zeros(AUDIO['audio_length_seconds'] * AUDIO['sample_rate'], dtype=np.int16)
        self._chunk_size = int(AUDIO['sample_rate']/AUDIO['update_frequency'])
        self._running = False
        # ignoring stdout during creation of PyAudio
        with ignoreStderr():
            p = pyaudio.PyAudio()
        self._stream = p.open(format=pyaudio.paInt16,
                        channels=AUDIO['channels'],
                        rate=AUDIO['sample_rate'],
                        input=True,
                        frames_per_buffer=self._chunk_size)
        device_info = p.get_default_input_device_info()
        log.info(f"AudioProvider initiated. \r\nSelected device: index {device_info['index']}, name: '{device_info['name']}', sample rate: {AUDIO['sample_rate']}")

    def _bytes_to_array(self, bytes):
        return np.frombuffer(bytes, np.int16)

    def stop(self):
        self._running = False
        self._stream.stop_stream()

    def read_audio_window(self):
        with self._lock:
            return self._audio_window

    def run(self):
        self._stream.start_stream()
        self._running = True
        while self._running:
            audio_bytes = self._stream.read(self._chunk_size)
            audio_arr = self._bytes_to_array(audio_bytes)
            tmp_audio = self._audio_window
            with self._lock:
                tmp_audio = [*tmp_audio, *audio_arr]
                del tmp_audio[:self._chunk_size]
                self._audio_window = tmp_audio
        self._stream.stop_stream()

def calculate_mfcc(audio_signal):
    """Returns Mel Frequency Cepstral Coefficients (MFCC) for a given audio signal.

    Args:
        audio_signal: Raw audio signal in range [-1, 1]

    Returns:
        Calculated mffc features.
    """

    window_size_in_seconds = MFCC['window_size']/AUDIO['sample_rate']
    window_stride_in_seconds = MFCC['window_stride']/AUDIO['sample_rate']
    mfcc_features = speech_features.mfcc(
        audio_signal,
        samplerate=AUDIO['sample_rate'],
        winlen=window_size_in_seconds,
        winstep=window_stride_in_seconds,
        numcep=MFCC['coefficient'],
        nfilt=40,
        lowfreq=20,
        highfreq=4000,
        nfft=MFCC['window_size'])
    mfcc_features = mfcc_features.astype(np.float32)

    return mfcc_features

def get_input_devices() -> dict:
    """Reads available input devices from ALSA. 

    Returns:
        dict: list of devices with all specifications, use index for AudioProvider()
    """
    p = pyaudio.PyAudio()
    device_list = []
    for d in range(p.get_device_count()):
        device_list.append(p.get_device_info_by_index(d))
    return device_list

@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)