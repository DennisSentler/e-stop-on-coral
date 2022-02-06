import logging
import argparse
import numpy as np
import pyaudio
import python_speech_features as speech_features
from test_tflite_tpu import tflite_inference
import audio_provider as audio_lib

TPU_ON = False
try:
    import tflite_runtime.interpreter as tflite_tpu
    logging.debug("Imported TPU tf lite interpreter succesfully")
    TPU_ON = True
except (ModuleNotFoundError, ImportError):
    from tensorflow import lite as tflite_cpu
    logging.debug("Imported CPU tf lite interpreter succesfully")

import sys
import matplotlib.pyplot as plt

def load_interpreter(tflite_path):
    if TPU_ON:
        interpreter = tflite_tpu.Interpreter(
                    tflite_path, 
                    experimental_delegates=[tflite_tpu.load_delegate('libedgetpu.so.1')]
            )
    else:
        interpreter = tflite_cpu.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()
    return interpreter

def calculate_mfcc(audio_signal, audio_sample_rate, window_size, window_stride, num_mfcc):
    """Returns Mel Frequency Cepstral Coefficients (MFCC) for a given audio signal.

    Args:
        audio_signal: Raw audio signal in range [-1, 1]
        audio_sample_rate: Audio signal sample rate
        window_size: Window size in samples for calculating spectrogram
        window_stride: Window stride in samples for calculating spectrogram
        num_mfcc: The number of MFCC features wanted.

    Returns:
        Calculated mffc features.
    """
    # spectrogram = audio_ops.audio_spectrogram(input=audio_signal, window_size=window_size, stride=window_stride,
    #                                           magnitude_squared=True)

    # mfcc_features = audio_ops.mfcc(spectrogram, audio_sample_rate, dct_coefficient_count=num_mfcc)

    window_size_in_seconds = window_size/audio_sample_rate
    window_stride_in_seconds = window_stride/audio_sample_rate
    mfcc_features = speech_features.mfcc(
        audio_signal,
        samplerate=audio_sample_rate,
        winlen=window_size_in_seconds,
        winstep=window_stride_in_seconds,
        numcep=num_mfcc,
        nfilt=40,
        lowfreq=20,
        highfreq=4000,
        nfft=window_size)
    mfcc_features = mfcc_features.astype(np.float32)

    return mfcc_features

def preProcessAudio(audio):
    audio_scaled = np.interp(audio, (np.iinfo(np.int16).min, np.iinfo(np.int16).max), (-1, +1))
    audio_mfcc = calculate_mfcc(audio_scaled, 16000, 640, 320, 10)
    audio_mfcc = np.reshape(audio_mfcc, (1,49,10))
    return audio_mfcc

def initAudioStream():
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    SAMPLE_RATE = 16000
    INPUT_LENGTH_MS = 1000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    stream.start_stream()
    return p, stream

def bytesToArray(bytes):
    return np.frombuffer(bytes, np.int16)

def saveDataPlot(audio, mfcc, name):
    fig, (audio_axis, mfcc_axis) = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(name)
    
    #audio plot
    audio_axis.set_ylim(np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    audio_axis.plot([*range(len(audio))], audio)
    audio_axis.set_xlabel("time")
    audio_axis.set_ylabel("a")

    #mfcc plot
    mfcc_axis.set_xlabel("mfcc coeffi.")
    mfcc_axis.set_xlabel("time")
    mfcc_axis.imshow(mfcc[0], interpolation='nearest', cmap='coolwarm', origin='lower')
    plt.savefig("plot.jpg")
    plt.cla()
    
# p, stream = initAudioStream()
audio_provider = audio_lib.AudioProvider()
audio_provider.start()
tpu = load_interpreter("models/compiled_for_edge_tpu/ds_cnn_quantized_v3_edgetpu.tflite")

# audio = []
# audio_bytes = stream.read(16000)
# audio.extend(bytesToArray(audio_bytes))

def inference():
    global audio_provider
    # chunk = int(16000/10)
    # audio_bytes = bytesToArray(stream.read(chunk)) 
    # tmp_audio = audio
    # tmp_audio.extend(audio_bytes)
    # del tmp_audio[:chunk]
    # audio = tmp_audio
    audio = audio_provider.read_audio_window()
    mfcc = preProcessAudio(audio)
    return tflite_inference(mfcc, tpu)[0]

def main():

    try:
        while True:
            predictions = inference()
            highes_pred = np.argmax(predictions)
            if predictions[highes_pred] > 0.7:
                print('Prediction: %s, with %dp' % (WORDS[highes_pred], predictions[highes_pred]*100))
    except (KeyboardInterrupt, SystemExit):
        # global stream
        # global p
        # stream.stop_stream()
        # p.close(stream)
        print("\nINFO: Applicatioon canceled!")
        sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--input_device',
        type=int,
        default=1,
        help='Index of the prefered device from pyaudio',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='models/compiled_for_edge_tpu/ds_cnn_q_2d_edgetpu.tflite',
        help='Path to TFLite file to use for testing. Must be compiled for TPU.'
        )
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)

    FLAGS, _ = parser.parse_known_args()
    WORDS = ["__silence__", "__unknown__"] + FLAGS.wanted_words.split(',')
    main()