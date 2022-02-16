import argparse
import numpy as np
import python_speech_features as speech_features
import audio_provider as audio_lib
from measure import clock
from config import MODEL
from logger import log

TPU_ON = False
CLOCK_INFERENCE = "Inference"
CLOCK_PRE_PROC = "Pre Processing"
CLOCK_TOTAL = "Total"

try:
    import tflite_runtime.interpreter as tflite_tpu
    log.info("Imported TPU tf lite interpreter succesfully")
    TPU_ON = True
except (ModuleNotFoundError, ImportError):
    from tensorflow import lite as tflite_cpu
    log.info("Imported CPU tf lite interpreter succesfully")

import sys
import matplotlib.pyplot as plt

class CommandDetector():
    def __init__(
            self,
            tflite_path: str, 
        ):
        clock.addClock(CLOCK_INFERENCE)
        clock.addClock(CLOCK_PRE_PROC)
        clock.addClock(CLOCK_TOTAL)
        self.audio_provider = audio_lib.AudioProvider()
        self.interpreter = load_interpreter(tflite_path)
        self.audio_provider.start()

    def inference(self):
        clock.start(CLOCK_TOTAL)
        audio = self.audio_provider.read_audio_window()
        clock.start(CLOCK_PRE_PROC)
        mfcc = self.preProcessAudio(audio)
        clock.stop(CLOCK_PRE_PROC)
        clock.start(CLOCK_INFERENCE)
        predictions = self.tflite_inference(mfcc)
        clock.stop(CLOCK_INFERENCE)
        clock.stop(CLOCK_TOTAL)
        return predictions

    def preProcessAudio(self, audio):
        audio_scaled = np.interp(audio, (np.iinfo(np.int16).min, np.iinfo(np.int16).max), (-1, +1))
        audio_mfcc = audio_lib.calculate_mfcc(audio_scaled)
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
    command_detector = CommandDetector(FLAGS.tflite_path)
    log.info("Inference starting ...")
    try:
        while True:
            predictions = command_detector.inference()
            highes_pred = np.argmax(predictions)
            if predictions[highes_pred] > 0.7:
                print('Prediction: %s, with %dp' % (MODEL['words'][highes_pred], predictions[highes_pred]*100))
    except (KeyboardInterrupt, SystemExit):
        log.info("Application shutdown")
        sys.exit()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='models/compiled_for_edge_tpu/ds_cnn_q_2d_edgetpu.tflite',
        help='Path to TFLite file to use for testing. Must be compiled for TPU.'
        )
    FLAGS, _ = parser.parse_known_args()
    main()