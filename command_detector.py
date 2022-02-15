import logging
logging.basicConfig(level=logging.INFO, format='LOG:%(asctime)s:%(levelname)s:%(name)s == %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging = logging.getLogger(__name__)

import argparse
import numpy as np
import python_speech_features as speech_features
import audio_provider as audio_lib
from measure import clockwatch

TPU_ON = False
INFERENCE = "Inference"
PRE_PROC = "Pre Processing"
try:
    import tflite_runtime.interpreter as tflite_tpu
    logging.info("Imported TPU tf lite interpreter succesfully")
    TPU_ON = True
except (ModuleNotFoundError, ImportError):
    from tensorflow import lite as tflite_cpu
    logging.info("Imported CPU tf lite interpreter succesfully")

import sys
import matplotlib.pyplot as plt

class CommandDetector():
    def __init__(
            self,
            command_list,
            tflite_path: str, 
            audio_sample_rate: int = 16000, 
            audio_window_ms: int = 1000,
            audio_reading_stride: int = 1000,
        ):
        clockwatch.addClock(INFERENCE)
        clockwatch.addClock(PRE_PROC)
        self.commands = command_list
        self.audio_provider = audio_lib.AudioProvider(
            sample_rate=audio_sample_rate, 
            audio_window_ms=audio_window_ms, 
            reading_stride=audio_reading_stride
        )
        self.interpreter = load_interpreter(tflite_path)
        self.audio_provider.start()


    def inference(self):
        audio = self.audio_provider.read_audio_window()
        clockwatch.start(PRE_PROC)
        mfcc = self.preProcessAudio(audio)
        clockwatch.stop(PRE_PROC)
        clockwatch.start(INFERENCE)
        predictions = self.tflite_inference(mfcc)
        clockwatch.stop(INFERENCE)
        return predictions

    def preProcessAudio(self, audio):
        audio_scaled = np.interp(audio, (np.iinfo(np.int16).min, np.iinfo(np.int16).max), (-1, +1))
        audio_mfcc = calculate_mfcc(audio_scaled, 16000, 640, 320, 10)
        audio_mfcc = np.reshape(audio_mfcc, (1,49,10))
        return audio_mfcc

    def tflite_inference(self, mfcc_input_data):
        """Call forwards pass of TFLite file and returns the result.

        Args:
            input_data: Input data to use on forward pass.
            interpreter: Intepreter from loaded tflite model.

        Returns:
            Output from inference.
        """
        interpreter = self.interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_dtype = input_details[0]["dtype"]
        output_dtype = output_details[0]["dtype"]

        # Check if the input/output type is quantized,
        # set scale and zero-point accordingly
        if input_dtype == np.int8:
            input_scale, input_zero_point = input_details[0]["quantization"]
        else:
            input_scale, input_zero_point = 1, 0

        mfcc_input_data = mfcc_input_data / input_scale + input_zero_point
        mfcc_input_data = np.round(mfcc_input_data) if input_dtype == np.int8 else mfcc_input_data

        if output_dtype == np.int8:
            output_scale, output_zero_point = output_details[0]["quantization"]
        else:
            output_scale, output_zero_point = 1, 0


        interpreter.set_tensor(input_details[0]['index'], mfcc_input_data.astype(input_dtype))
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)
        #get inner array
        output_data = output_data[0]
        return output_data

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

def main():
    audio_reading_stride = int(FLAGS.sample_rate / FLAGS.audio_access_per_second)
    
    command_detector = CommandDetector(
        WORDS, 
        FLAGS.tflite_path,
        FLAGS.sample_rate, 
        FLAGS.clip_duration_ms,
        audio_reading_stride, 
    )
    logging.info("Inference starting ...")
    try:
        while True:
            predictions = command_detector.inference()
            highes_pred = np.argmax(predictions)
            if predictions[highes_pred] > 0.7:
                print('Prediction: %s, with %dp' % (WORDS[highes_pred], predictions[highes_pred]*100))
    except (KeyboardInterrupt, SystemExit):
        logging.info("Application shutdown")
        sys.exit()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the audio',)
    parser.add_argument(
        '--audio_access_per_second',
        type=int,
        default=10,
        help='Number of audio accesses per second',)
    parser.add_argument(
        '--mfcc_window_size_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--mfcc_window_stride_ms',
        type=float,
        default=20.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--mfcc_coefficient_count',
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