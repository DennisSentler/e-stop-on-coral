import argparse
import numpy as np
import audio_provider as Audio
import interpreter
import sys

def main():
    mic = Audio.AudioProvider(FLAGS.input_device)
    mic.start()
    tpu = interpreter.Intepreter(FLAGS.tflite_path, name="Test")
    tpu.start()

    try:
        while True:
            audio = mic.read_audio_window()
            mfccs = Audio.pre_proc_audio(audio)
            tpu.set_input(mfccs)
            prediction = tpu.get_output()
            if None not in prediction:
                print(np.argmax(prediction))

    except (KeyboardInterrupt, SystemExit):
        mic.stop()
        tpu.stop()
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
        default=6,
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
    #WORDS = data_processor.prepare_words_list(FLAGS.wanted_words.split(','))
    main()